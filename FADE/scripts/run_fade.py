from pathlib import Path

import click
import cv2
import gem
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from datasets.base import IMAGENET_MEAN, IMAGENET_STD
from utils.anomaly_detection import predict_classification, predict_segmentation
from utils.hdf5_io import load_hdf5_input
from utils.embeddings import extract_image_embeddings, retrieve_image_embeddings
from utils.image_model import (
    build_image_models,
    extract_query_patch_embeddings,
    combine_patch_embeddings,
)
from utils.text_model import build_text_model


def preprocess_single_image(image_or_path, img_sizes: list, square: bool = True) -> dict:
    if isinstance(image_or_path, (str, Path)):
        image = Image.open(image_or_path).convert("RGB")
    else:
        image = image_or_path.convert("RGB")
    result = {}
    for sz in img_sizes:
        t = transforms.Compose([
            transforms.Resize(
                (sz, sz) if square else sz,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        result[sz] = t(image).unsqueeze(0)
    return result


def extract_embeddings_from_paths(
    image_paths: list,
    img_sizes: list,
    gem_model,
    feature_type: str,
    device: str,
    square: bool = True,
) -> dict:
    patch_embeddings = {sz: [] for sz in img_sizes}
    for path in image_paths:
        if str(path).lower().endswith(".hdf5"):
            pil_ref, _, _ = load_hdf5_input(str(path))
            multiscale = preprocess_single_image(pil_ref, img_sizes, square)
        else:
            multiscale = preprocess_single_image(path, img_sizes, square)
        embs = extract_image_embeddings(multiscale, gem_model, device)
        for sz in img_sizes:
            patch_embeddings[sz].append(
                retrieve_image_embeddings(embs, img_size=sz, feature_type=feature_type, token_type="patch")
            )
    return {sz: np.concatenate(patch_embeddings[sz]) for sz in img_sizes}


def _instance_iou(gt_labeled: np.ndarray, pred_labeled: np.ndarray, gt_id: int, pred_id: int) -> float:
    gt_blob   = gt_labeled == gt_id
    pred_blob = pred_labeled == pred_id
    intersection = (gt_blob & pred_blob).sum()
    union        = (gt_blob | pred_blob).sum()
    return intersection / union if union > 0 else 0.0


def _instance_precision_recall(gt_binary: np.ndarray, pred_binary: np.ndarray, iou_threshold: float = 0.5):
    from scipy.ndimage import label as ndlabel

    gt_labeled,   n_gt   = ndlabel(gt_binary)
    pred_labeled, n_pred = ndlabel(pred_binary)

    if n_gt == 0 and n_pred == 0:
        return 1.0, 1.0, n_gt, n_pred
    if n_gt == 0:
        return 0.0, 1.0, n_gt, n_pred
    if n_pred == 0:
        return 1.0, 0.0, n_gt, n_pred

    matched_gt   = set()
    matched_pred = set()
    for g in range(1, n_gt + 1):
        for p in range(1, n_pred + 1):
            if p in matched_pred:
                continue
            if _instance_iou(gt_labeled, pred_labeled, g, p) >= iou_threshold:
                matched_gt.add(g)
                matched_pred.add(p)
                break

    inst_precision = len(matched_pred) / n_pred
    inst_recall    = len(matched_gt)   / n_gt
    return inst_precision, inst_recall, n_gt, n_pred

def compute_metrics(seg_float: np.ndarray, gt_mask: np.ndarray, ignore_mask: np.ndarray | None, final_score: float):
    from sklearn.metrics import roc_auc_score, f1_score, jaccard_score, precision_score, recall_score

    # Adjust these two thresholds to tune detection sensitivity
    PIX_THRESHOLD = 0.3  # lower = more pixels flagged as anomaly
    IMG_THRESHOLD = 0.3 # lower = more images classified as anomaly

    eval_size = seg_float.shape[0]
    gt_resized = cv2.resize(gt_mask.astype(np.uint8), (eval_size, eval_size), interpolation=cv2.INTER_NEAREST)
    gt_binary = (gt_resized > 0).astype(np.uint8)

    if ignore_mask is not None:
        ignore_resized = cv2.resize(ignore_mask.astype(np.uint8), (eval_size, eval_size), interpolation=cv2.INTER_NEAREST)
        valid_mask = (ignore_resized == 0)
    else:
        valid_mask = np.ones((eval_size, eval_size), dtype=bool)

    pred_flat = seg_float[valid_mask]
    gt_flat   = gt_binary[valid_mask]

    print("\n=== Accuracy Metrics ===")

    gt_img_label = int(gt_mask.max() > 0)
    pred_img_label = int(final_score > IMG_THRESHOLD)
    print(f"Image-level GT : {'anomaly' if gt_img_label else 'normal'}")
    print(f"Image-level pred: {'anomaly' if pred_img_label else 'normal'} (score={final_score:.4f}, threshold={IMG_THRESHOLD})")

    n_pos = int(gt_flat.sum())
    n_neg = len(gt_flat) - n_pos
    if n_pos == 0 or n_neg == 0:
        print("Pixel AUROC    : N/A (mask is all-one or all-zero in valid region)")
    else:
        auroc = roc_auc_score(gt_flat, pred_flat)
        print(f"Pixel AUROC    : {auroc:.4f}")

    pred_binary_flat = (pred_flat > PIX_THRESHOLD).astype(np.uint8)
    f1        = f1_score(gt_flat, pred_binary_flat, zero_division=0)
    iou       = jaccard_score(gt_flat, pred_binary_flat, zero_division=0)
    precision = precision_score(gt_flat, pred_binary_flat, zero_division=0)
    recall    = recall_score(gt_flat, pred_binary_flat, zero_division=0)
    n_pred_pixels = int(pred_binary_flat.sum())
    print(f"--- Pixel level (threshold={PIX_THRESHOLD}) ---")
    print(f"  predicted anomaly pixels : {n_pred_pixels} / {len(gt_flat)}")
    print(f"  GT anomaly pixels        : {n_pos} / {len(gt_flat)}")
    if n_pred_pixels == 0:
        print(f"  (no pixels predicted — threshold may be too high)")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"IoU       : {iou:.4f}")
    print(f"  (anomaly pixels in valid region: {n_pos} / {len(gt_flat)})")

    defect_scores = pred_flat[gt_flat == 1]
    if len(defect_scores) > 0:
        print(f"Defect pixel scores — min: {defect_scores.min():.3f}, max: {defect_scores.max():.3f}, mean: {defect_scores.mean():.3f}")
    # Instance-level: apply valid_mask to both maps before finding blobs
    gt_binary_valid   = gt_binary   * valid_mask
    pred_binary_valid = (seg_float > PIX_THRESHOLD).astype(np.uint8) * valid_mask
    inst_p, inst_r, n_gt, n_pred = _instance_precision_recall(gt_binary_valid, pred_binary_valid, iou_threshold=0.5)
    print(f"--- Instance level (pixel threshold={PIX_THRESHOLD}, IoU threshold=0.5) ---")
    print(f"Precision : {inst_p:.4f}  ({int(inst_p * n_pred)}/{n_pred} predicted instances matched a GT defect)")
    print(f"Recall    : {inst_r:.4f}  ({int(inst_r * n_gt)}/{n_gt} GT defect instances found)")


@click.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--classname", type=str, default="object",
              help="Object class name for text prompts (e.g. 'carpet').")
@click.option("--ref-images", type=str, default=None,
              help="Comma-separated paths to reference 'good' images (.png/.jpg or .hdf5 with 'img').")
@click.option("--experiment-name", type=str, default="output",
              help="Directory where outputs are saved.")
@click.option("--model-name", type=str, default="ViT-B/16-plus-240")
@click.option("--pretrained", type=str,
              default="models/openclip/clip/vit_b_16_plus_240-laion400m_e31-8fb26589.pt")
@click.option("--language-classification-feature", type=click.Choice(["clip", "gem"]), default="clip")
@click.option("--language-segmentation-feature", type=click.Choice(["clip", "gem"]), default="gem")
@click.option("--vision-feature", type=click.Choice(["clip", "gem"]), default="gem")
@click.option("--vision-segmentation-multiplier", type=float, default=3.5)
@click.option("--vision-segmentation-weight", type=click.FloatRange(0.0, 1.0), default=0.85)
@click.option("--use-query-img-in-vision-memory-bank/--no-use-query-img-in-vision-memory-bank",
              type=bool, default=False)
@click.option("--classification-img-size", type=int, default=240)
@click.option("--segmentation-img-sizes", type=str, default="240,448,896")
@click.option("--eval-img-size", type=int, default=448)
@click.option("--square/--no-square", type=bool, default=True)
@click.option("--text-model-type",
              type=click.Choice(["average", "softmax", "max", "lr", "mlp", "knn", "rf", "xgboost", "gmm"]),
              default="average")
@click.option("--normalize-segmentations/--no-normalize-segmentations", type=bool, default=False)
def main(
    image_path: str,
    classname: str,
    ref_images: str,
    experiment_name: str,
    model_name: str,
    pretrained: str,
    language_classification_feature: str,
    language_segmentation_feature: str,
    vision_feature: str,
    vision_segmentation_multiplier: float,
    vision_segmentation_weight: float,
    use_query_img_in_vision_memory_bank: bool,
    classification_img_size: int,
    segmentation_img_sizes: str,
    eval_img_size: int,
    square: bool,
    text_model_type: str,
    normalize_segmentations: bool,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_sizes = [int(i) for i in segmentation_img_sizes.split(",")]

    # Load image — HDF5 files are read directly; PNG/JPEG go through the normal path
    pil_image = None
    gt_mask = None
    ignore_mask = None
    if str(image_path).lower().endswith(".hdf5"):
        print(f"Loading image from HDF5: {image_path}")
        pil_image, gt_mask, ignore_mask = load_hdf5_input(image_path)
        if gt_mask is not None:
            print(f"  Ground truth mask loaded (anomaly pixels: {int(gt_mask.max() > 0 and gt_mask.sum())})")
    else:
        pil_image = Image.open(image_path).convert("RGB")

    text_prompt_path = "prompts"
    prompt_paths_classification = [f"{text_prompt_path}/winclip_prompt.json"]
    prompt_paths_segmentation = [
        f"{text_prompt_path}/winclip_prompt.json",
        f"{text_prompt_path}/chatgpt3.5_prompt1.json",
        f"{text_prompt_path}/chatgpt3.5_prompt2.json",
        f"{text_prompt_path}/chatgpt3.5_prompt3.json",
        f"{text_prompt_path}/chatgpt3.5_prompt4.json",
        f"{text_prompt_path}/chatgpt3.5_prompt5.json",
    ]

    print(f"Loading GEM model ({model_name})...")
    gem_model = gem.create_gem_model(model_name=model_name, pretrained=pretrained, device=device)
    if device == "cuda":
        gem_model.model = gem_model.model.half()
        gem_model.model.visual = torch.compile(gem_model.model.visual)

    print("Building text models...")
    classname_display = classname.replace("_", " ")
    classification_text_model = build_text_model(
        gem_model=gem_model,
        prompt_paths=prompt_paths_classification,
        classname=classname_display,
        text_model_type=text_model_type,
    )
    segmentation_text_model = build_text_model(
        gem_model=gem_model,
        prompt_paths=prompt_paths_segmentation,
        classname=classname_display,
        text_model_type=text_model_type,
    )

    print(f"Processing image: {image_path}")
    all_sizes = list({classification_img_size, *seg_sizes, eval_img_size})
    multiscale_images = preprocess_single_image(pil_image, all_sizes, square)
    image_embeddings = extract_image_embeddings(multiscale_images, gem_model, device)

    # Language-guided branch
    language_scores = predict_classification(
        text_model=classification_text_model,
        image_embeddings=image_embeddings,
        img_size=classification_img_size,
        feature_type=language_classification_feature,
    )
    language_maps = predict_segmentation(
        model=segmentation_text_model,
        image_embeddings=image_embeddings,
        img_sizes=seg_sizes,
        feature_type=language_segmentation_feature,
        patch_size=gem_model.model.visual.patch_size,
        segmentation_mode="language",
    )

    # Vision-guided branch (optional, needs reference images or self-memory)
    ref_image_paths = [p.strip() for p in ref_images.split(",")] if ref_images else []
    use_vision = bool(ref_image_paths) or use_query_img_in_vision_memory_bank

    vision_scores = None
    vision_maps = None
    if use_vision:
        ref_patch_embeddings = None
        if ref_image_paths:
            print(f"Building vision memory bank from {len(ref_image_paths)} reference image(s)...")
            ref_patch_embeddings = extract_embeddings_from_paths(
                ref_image_paths, seg_sizes, gem_model, vision_feature, device, square
            )
        query_patch_embeddings = None
        if use_query_img_in_vision_memory_bank:
            query_patch_embeddings = extract_query_patch_embeddings(
                image_embeddings, seg_sizes, vision_feature
            )
        if ref_patch_embeddings is not None and query_patch_embeddings is not None:
            train_patch_embeddings = combine_patch_embeddings(ref_patch_embeddings, query_patch_embeddings)
        else:
            train_patch_embeddings = ref_patch_embeddings or query_patch_embeddings

        image_models = build_image_models(train_patch_embeddings, use_query_img_in_vision_memory_bank)
        vision_maps = predict_segmentation(
            model=image_models,
            image_embeddings=image_embeddings,
            img_sizes=seg_sizes,
            feature_type=vision_feature,
            patch_size=gem_model.model.visual.patch_size,
            segmentation_mode="vision",
        )
        vision_maps *= vision_segmentation_multiplier
        vision_scores = np.max(vision_maps, axis=(1, 2))

    # Fuse language and vision results
    if use_vision:
        final_score = float((language_scores[0] + vision_scores[0]) / 2)
        final_map = (
            (1.0 - vision_segmentation_weight) * language_maps
            + vision_segmentation_weight * vision_maps
        )
    else:
        final_score = float(language_scores[0])
        final_map = language_maps

    # Post-process segmentation
    seg = final_map[0]
    if normalize_segmentations:
        lo, hi = seg.min(), seg.max()
        seg = (seg - lo) / (hi - lo + 1e-8)
    seg = np.clip(seg, 0, 1)
    # Resize as float first for higher quality and to keep scores for metrics
    seg_float_resized = cv2.resize(seg.astype(np.float32), (eval_img_size, eval_img_size))
    seg_resized = (seg_float_resized * 255).astype("uint8")

    # Save outputs
    out_dir = Path(experiment_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    seg_path = out_dir / f"{stem}_segmentation.png"
    Image.fromarray(seg_resized).save(seg_path)

    heatmap = cv2.applyColorMap(seg_resized, cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    original_resized = cv2.resize(original_bgr, (eval_img_size, eval_img_size))
    overlay = cv2.addWeighted(original_resized, 0.5, heatmap, 0.5, 0)
    overlay_path = out_dir / f"{stem}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    print(f"\n=== Results ===")
    print(f"Classification score : {final_score:.4f}  (0 = normal, 1 = anomaly)")
    print(f"Segmentation map     : {seg_path}")
    print(f"Heatmap overlay      : {overlay_path}")

    if gt_mask is not None:
        compute_metrics(seg_float_resized, gt_mask, ignore_mask, final_score)

if __name__ == "__main__":
    main()
