"""Shared HDF5 image / mask loading for FADE scripts."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def load_hdf5_input(hdf5_path: str | Path):
    """
    Load RGB image and optional ground-truth / ignore masks from an HDF5 file.

    Expected keys: ``img`` (required), ``mask`` (optional, anomaly > 0),
    ``ignore_mask`` (optional, non-zero = excluded from evaluation / clean patches).
    """
    gt_mask = None
    ignore_mask = None
    with h5py.File(hdf5_path, "r") as f:
        img_array = np.asarray(f["img"][:])
        if "mask" in f:
            gt_mask = np.asarray(f["mask"][:])
        if "ignore_mask" in f:
            ignore_mask = np.asarray(f["ignore_mask"][:])

    img = img_array
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=-1)

    if img.dtype != np.uint8:
        max_val = float(np.max(img)) if img.size else 1.0
        if max_val <= 0:
            max_val = 1.0
        img = (img.astype(np.float64) / max_val * 255.0).clip(0, 255).astype(np.uint8)

    pil_image = Image.fromarray(img, mode="RGB")
    return pil_image, gt_mask, ignore_mask


def align_mask_to_image(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize mask to (height, width) of ``target_hw`` using nearest neighbour."""
    th, tw = target_hw
    if mask.shape[0] == th and mask.shape[1] == tw:
        return mask
    im = Image.fromarray(np.asarray(mask), mode="L")
    im = im.resize((tw, th), resample=Image.Resampling.NEAREST)
    return np.asarray(im)
