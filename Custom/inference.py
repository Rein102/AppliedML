import os
import time
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


ROOT = ""

TRAIN_LABELLED = f""
TEST_UNLABELLED = f""

OUT_DIR = ""
os.makedirs(OUT_DIR, exist_ok=True)

PATCH_SIZE = 16

MAX_CLEAN_PATCHES = 20_000
N_NEIGHBORS = 5

RED_PERCENTILE = 99.75
YELLOW_PERCENTILE = 99.4

COMPONENT_GRAY_PERCENTILE = 85
COMPONENT_EDGE_PERCENTILE = 80
EXTREME_SCORE_PERCENTILE = 99.95

MIN_AREA = 3


def list_hdf5(root):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".hdf5"):
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)


def domain_from_name(path):
    name = os.path.basename(path)
    if "red19" in name:
        return "red19"
    if "yellow12" in name:
        return "yellow12"
    return "unknown"


def make_feature_maps(img):
    img_f = img.astype(np.float32)

    r = img_f[..., 0]
    g = img_f[..., 1]
    b = img_f[..., 2]

    rgb_sum = r + g + b + 1.0

    rn = r / rgb_sum
    gn = g / rgb_sum
    bn = b / rgb_sum

    purple_ratio = (r + b) / (g + 1.0)
    rg = r - g
    bg = b - g

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    local_contrast = np.abs(gray - blur)

    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))

    maps = np.stack(
        [
            rn,
            gn,
            bn,
            purple_ratio,
            rg,
            bg,
            lab[..., 1],
            lab[..., 2],
            hsv[..., 0],
            hsv[..., 1],
            hsv[..., 2],
            gray,
            local_contrast,
            lap,
        ],
        axis=-1,
    )

    return maps.astype(np.float32)


def patch_stats_from_maps(feature_maps, patch_shape):
    ph, pw = patch_shape
    h = ph * PATCH_SIZE
    w = pw * PATCH_SIZE

    feature_maps = feature_maps[:h, :w]

    x = feature_maps.reshape(
        ph,
        PATCH_SIZE,
        pw,
        PATCH_SIZE,
        feature_maps.shape[-1],
    )

    x = x.transpose(0, 2, 1, 3, 4)

    mean = x.mean(axis=(2, 3))
    std = x.std(axis=(2, 3))

    return np.concatenate([mean, std], axis=-1).astype(np.float32)


def extract_patch_features_from_file(path, only_clean=False):
    with h5py.File(path, "r") as f:
        img = f["img"][:]
        patch_mask = f["patch_mask"][:]
        patch_ignore = f["patch_ignore_mask"][:]

    feature_maps = make_feature_maps(img)
    features_map = patch_stats_from_maps(feature_maps, patch_mask.shape)

    valid = patch_ignore == 0
    labels = (patch_mask > 0).astype(np.uint8)

    if only_clean:
        selector = valid & (labels == 0)
    else:
        selector = valid

    features = features_map[selector]

    return features.astype(np.float32)


def collect_clean_features(paths):
    rng = np.random.default_rng(42)
    all_features = []

    for path in tqdm(paths, desc="Collect clean features"):
        features = extract_patch_features_from_file(path, only_clean=True)
        all_features.append(features)

    all_features = np.concatenate(all_features, axis=0)

    if len(all_features) > MAX_CLEAN_PATCHES:
        idx = rng.choice(
            len(all_features),
            size=MAX_CLEAN_PATCHES,
            replace=False,
        )
        all_features = all_features[idx]

    return all_features.astype(np.float32)


def fit_model(clean_features):
    scaler = StandardScaler()
    clean_scaled = scaler.fit_transform(clean_features)

    nn = NearestNeighbors(
        n_neighbors=N_NEIGHBORS,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    )

    nn.fit(clean_scaled)

    return {
        "scaler": scaler,
        "nn": nn,
    }


def score_feature_map(features_map, model):
    scaler = model["scaler"]
    nn = model["nn"]

    ph, pw, d = features_map.shape

    flat = features_map.reshape(-1, d)
    flat_scaled = scaler.transform(flat)

    distances, _ = nn.kneighbors(flat_scaled)
    scores = distances[:, -1]

    return scores.reshape(ph, pw).astype(np.float32)


def choose_domain_and_scores(features_map, models):
    domain_scores = {}
    score_maps = {}

    for domain, model in models.items():
        score_map = score_feature_map(features_map, model)
        score_maps[domain] = score_map
        domain_scores[domain] = float(np.mean(score_map))

    chosen_domain = min(domain_scores, key=domain_scores.get)

    return chosen_domain, domain_scores, score_maps


def make_component_mask(features_map, ignore_map):
    gray_mean = features_map[..., 11]
    contrast_mean = features_map[..., 12]
    lap_mean = features_map[..., 13]

    valid = ignore_map == 0

    gray_thr = np.percentile(gray_mean[valid], COMPONENT_GRAY_PERCENTILE)
    contrast_thr = np.percentile(contrast_mean[valid], COMPONENT_EDGE_PERCENTILE)
    lap_thr = np.percentile(lap_mean[valid], COMPONENT_EDGE_PERCENTILE)

    bright = gray_mean > gray_thr
    structured = (contrast_mean > contrast_thr) | (lap_mean > lap_thr)

    component_mask = bright & structured
    component_mask[ignore_map > 0] = 0

    return component_mask.astype(np.uint8)


def remove_small_components(pred_map, min_area=3):
    pred = pred_map.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        pred,
        connectivity=8,
    )

    cleaned = np.zeros_like(pred)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == i] = 1

    return cleaned


def save_unlabelled_visualization(
    path,
    img,
    score_map,
    component_mask,
    pred_map,
    chosen_domain,
    inference_time,
):
    name = os.path.basename(path).replace(".hdf5", "")

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(score_map, cmap="jet")
    axes[1].set_title(f"Anomaly score ({chosen_domain})")
    axes[1].axis("off")

    axes[2].imshow(component_mask, cmap="gray")
    axes[2].set_title("Component mask")
    axes[2].axis("off")

    axes[3].imshow(pred_map, cmap="Reds")
    axes[3].set_title("Prediction")
    axes[3].axis("off")

    plt.suptitle(f"{name} | inference {inference_time:.2f}s")
    plt.tight_layout()

    out_path = os.path.join(
        OUT_DIR,
        name + f"_domain_{chosen_domain}_pred.png",
    )

    plt.savefig(out_path, dpi=150)
    plt.close()

    print("saved:", out_path)


def infer_one_file(path, models):
    start = time.perf_counter()

    with h5py.File(path, "r") as f:
        img = f["img"][:]

        if "patch_ignore_mask" in f:
            patch_ignore = f["patch_ignore_mask"][:]
        else:
            h, w = img.shape[:2]
            patch_ignore = np.zeros(
                (h // PATCH_SIZE, w // PATCH_SIZE),
                dtype=np.uint8,
            )

    feature_maps = make_feature_maps(img)
    features_map = patch_stats_from_maps(feature_maps, patch_ignore.shape)

    chosen_domain, domain_scores, score_maps = choose_domain_and_scores(
        features_map,
        models,
    )

    score_map = score_maps[chosen_domain]

    valid_scores = score_map[patch_ignore == 0]

    if chosen_domain == "red19":
        threshold = np.percentile(valid_scores, RED_PERCENTILE)
    elif chosen_domain == "yellow12":
        threshold = np.percentile(valid_scores, YELLOW_PERCENTILE)
    else:
        threshold = np.percentile(valid_scores, 99.8)

    pred_map = (score_map > threshold).astype(np.uint8)
    pred_map[patch_ignore > 0] = 0

    component_mask = make_component_mask(features_map, patch_ignore)

    extreme_threshold = np.percentile(valid_scores, EXTREME_SCORE_PERCENTILE)

    suppress_mask = (component_mask == 1) & (score_map < extreme_threshold)
    pred_map[suppress_mask] = 0

    pred_map = remove_small_components(pred_map, min_area=MIN_AREA)

    elapsed = time.perf_counter() - start

    return {
        "img": img,
        "score_map": score_map,
        "component_mask": component_mask,
        "pred_map": pred_map,
        "chosen_domain": chosen_domain,
        "domain_scores": domain_scores,
        "threshold": threshold,
        "inference_time": elapsed,
    }


def main():
    train_files = list_hdf5(TRAIN_LABELLED)
    all_unlabelled_files = list_hdf5(TEST_UNLABELLED)

    unlabelled_files = [
        p for p in all_unlabelled_files
        if "Die" in os.path.basename(p)
    ]

    red_train = [p for p in train_files if domain_from_name(p) == "red19"]
    yellow_train = [p for p in train_files if domain_from_name(p) == "yellow12"]

    print("red train files:", len(red_train))
    print("yellow train files:", len(yellow_train))
    print("unlabelled test files:", len(unlabelled_files))

    models = {}

    print("\nTraining red19 model")
    red_features = collect_clean_features(red_train)
    print("red clean features:", red_features.shape)
    models["red19"] = fit_model(red_features)

    print("\nTraining yellow12 model")
    yellow_features = collect_clean_features(yellow_train)
    print("yellow clean features:", yellow_features.shape)
    models["yellow12"] = fit_model(yellow_features)

    inference_times = []

    print("\nRunning inference on unlabelled files")

    for path in tqdm(unlabelled_files):
        result = infer_one_file(path, models)

        inference_times.append(result["inference_time"])

        print()
        print(os.path.basename(path))
        print("chosen domain:", result["chosen_domain"])
        print("domain scores:", result["domain_scores"])
        print("threshold:", result["threshold"])
        print("inference time:", result["inference_time"])

        save_unlabelled_visualization(
            path,
            result["img"],
            result["score_map"],
            result["component_mask"],
            result["pred_map"],
            result["chosen_domain"],
            result["inference_time"],
        )

    inference_times = np.array(inference_times)

    print("\n=== Inference timing ===")
    print("num images:", len(inference_times))
    print("mean seconds:", float(inference_times.mean()))
    print("median seconds:", float(np.median(inference_times)))
    print("min seconds:", float(inference_times.min()))
    print("max seconds:", float(inference_times.max()))
    print("outputs:", OUT_DIR)


if __name__ == "__main__":
    main()
