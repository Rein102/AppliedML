import sys
import json
from pathlib import Path

import click
import cv2
import gem
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.base import IMAGENET_MEAN, IMAGENET_STD
from utils.anomaly_detection import predict_classification, predict_segmentation
from utils.embeddings import extract_image_embeddings, retrieve_image_embeddings
from utils.image_model import build_image_models, extract_query_patch_embeddings, combine_patch_embeddings
from utils.text_model import build_text_model


def preprocess_single_image(image_path, img_sizes, square=True):
    image = Image.open(image_path).convert("RGB")
    result = {}
    for sz in img_sizes:
        t = transforms.Compose([
            transforms.Resize((sz, sz) if square else sz, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        result[sz] = t(image).unsqueeze(0)
    return result


def extract_embeddings_from_paths(image_paths, img_sizes, gem_model, feature_type, device, square=True):
    patch_embeddings = {sz: [] for sz in img_sizes}
    for path in image_paths:
        multiscale = preprocess_single_image(path, img_sizes, square)
        embs = extract_image_embeddings(multiscale, gem_model, device)
        for sz in img_sizes:
            patch_embeddings[sz].append(
                retrieve_image_embeddings(embs, img_size=sz, feature_type=feature_type, token_type="patch")
            )
    return {sz: np.concatenate(patch_embeddings[sz]) for sz in img_sizes}


def process_image(
    image_path, experiment_name, ref_image_paths,
    gem_model, classification_text_model, segmentation_text_model,
    seg_sizes, classification_img_size, eval_img_size,
    language_classification_feature, language_segmentation_feature, vision_feature,
    vision_segmentation_multiplier, vision_segmentation_weight,
    use_query_img_in_vision_memory_bank, square, normalize_segmentations, device,
):
    all_sizes = list({classification_img_size, *seg_sizes, eval_img_size})
    multiscale_images = preprocess_single_image(image_path, all_sizes, square)
    image_embeddings = extract_image_embeddings(multiscale_images, gem_model, device)

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

    use_vision = bool(ref_image_paths) or use_query_img_in_vision_memory_bank
    vision_scores = vision_maps = None
    if use_vision:
        ref_patch_embeddings = None
        if ref_image_paths:
            ref_patch_embeddings = extract_embeddings_from_paths(
                ref_image_paths, seg_sizes, gem_model, vision_feature, device, square
            )
        query_patch_embeddings = None
        if use_query_img_in_vision_memory_bank:
            query_patch_embeddings = extract_query_patch_embeddings(image_embeddings, seg_sizes, vision_feature)
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

    if use_vision:
        final_score = float((language_scores[0] + vision_scores[0]) / 2)
        final_map = (1.0 - vision_segmentation_weight) * language_maps + vision_segmentation_weight * vision_maps
    else:
        final_score = float(language_scores[0])
        final_map = language_maps

    seg = final_map[0]
    if normalize_segmentations:
        lo, hi = seg.min(), seg.max()
        seg = (seg - lo) / (hi - lo + 1e-8)
    seg = np.clip(seg, 0, 1)
    seg_uint8 = (seg * 255).astype("uint8")
    seg_resized = cv2.resize(seg_uint8, (eval_img_size, eval_img_size))

    out_dir = Path(experiment_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    seg_path = out_dir / f"{stem}_segmentation.png"
    Image.fromarray(seg_resized).save(seg_path)

    heatmap = cv2.applyColorMap(seg_resized, cv2.COLORMAP_JET)
    original = cv2.imread(str(image_path))
    original_resized = cv2.resize(original, (eval_img_size, eval_img_size))
    overlay = cv2.addWeighted(original_resized, 0.5, heatmap, 0.5, 0)
    overlay_path = out_dir / f"{stem}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    return final_score, seg_path, overlay_path


@click.command()
@click.option("--classname", type=str, default="object")
@click.option("--model-name", type=str, default="ViT-B/16-plus-240")
@click.option("--pretrained", type=str, default="models/openclip/clip/vit_b_16_plus_240-laion400m_e31-8fb26589.pt")
@click.option("--language-classification-feature", type=click.Choice(["clip", "gem"]), default="clip")
@click.option("--language-segmentation-feature", type=click.Choice(["clip", "gem"]), default="gem")
@click.option("--vision-feature", type=click.Choice(["clip", "gem"]), default="gem")
@click.option("--vision-segmentation-multiplier", type=float, default=3.5)
@click.option("--vision-segmentation-weight", type=click.FloatRange(0.0, 1.0), default=0.85)
@click.option("--use-query-img-in-vision-memory-bank/--no-use-query-img-in-vision-memory-bank", default=False)
@click.option("--classification-img-size", type=int, default=240)
@click.option("--segmentation-img-sizes", type=str, default="240")
@click.option("--eval-img-size", type=int, default=240)
@click.option("--square/--no-square", default=True)
@click.option("--text-model-type", type=click.Choice(["average", "softmax", "max", "lr", "mlp", "knn", "rf", "xgboost", "gmm"]), default="average")
@click.option("--normalize-segmentations/--no-normalize-segmentations", default=False)
@click.option("--port", type=int, default=5000, help="Port for the socket server")
def main(classname, model_name, pretrained, language_classification_feature, language_segmentation_feature,
         vision_feature, vision_segmentation_multiplier, vision_segmentation_weight,
         use_query_img_in_vision_memory_bank, classification_img_size, segmentation_img_sizes,
         eval_img_size, square, text_model_type, normalize_segmentations, port):

    import socket

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_sizes = [int(i) for i in segmentation_img_sizes.split(",")]

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

    print(f"Loading GEM model ({model_name}) on {device}...")
    gem_model = gem.create_gem_model(model_name=model_name, pretrained=pretrained, device=device)
    if device == "cuda":
        gem_model.model = gem_model.model.half()

    print("Building text models...")
    classification_text_model = build_text_model(
        gem_model=gem_model, prompt_paths=prompt_paths_classification,
        classname=classname.replace("_", " "), text_model_type=text_model_type,
    )
    segmentation_text_model = build_text_model(
        gem_model=gem_model, prompt_paths=prompt_paths_segmentation,
        classname="object", text_model_type=text_model_type,
    )

    print(f"\nModel ready. Listening on port {port}...")
    print("Send JSON: {\"image\": \"path\", \"experiment\": \"output/zero-shot\", \"ref_images\": []}")

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", port))
    server_sock.listen(5)

    while True:
        conn, _ = server_sock.accept()
        try:
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
            req = json.loads(data.decode())
            image_path = req["image"]
            experiment_name = req.get("experiment", "output/zero-shot")
            ref_image_paths = req.get("ref_images", [])

            print(f"Processing: {image_path}")
            score, seg_path, overlay_path = process_image(
                image_path, experiment_name, ref_image_paths,
                gem_model, classification_text_model, segmentation_text_model,
                seg_sizes, classification_img_size, eval_img_size,
                language_classification_feature, language_segmentation_feature, vision_feature,
                vision_segmentation_multiplier, vision_segmentation_weight,
                use_query_img_in_vision_memory_bank, square, normalize_segmentations, device,
            )
            result = {"score": score, "segmentation": str(seg_path), "overlay": str(overlay_path)}
            print(f"Score: {score:.4f}")
            conn.sendall(json.dumps(result).encode())
        except Exception as e:
            conn.sendall(json.dumps({"error": str(e)}).encode())
        finally:
            conn.close()


if __name__ == "__main__":
    main()
