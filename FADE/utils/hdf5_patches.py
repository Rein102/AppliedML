"""Extract clean (non-anomaly) image patches from labelled HDF5 volumes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from utils.hdf5_io import align_mask_to_image, load_hdf5_input


def extract_clean_patches_from_hdf5(
    hdf5_path: str | Path,
    output_dir: str | Path,
    *,
    patch_size: int = 128,
    stride: int | None = None,
    max_patches: int | None = None,
    exclude_ignore_regions: bool = True,
    filename_prefix: str | None = None,
) -> list[Path]:
    """
    Save square patches that contain no anomaly pixels according to ``mask``.

    Uses the same HDF5 layout as ``load_hdf5_input``: ``img``, optional ``mask``
    (anomaly where values > 0), optional ``ignore_mask`` (non-zero pixels are
    skipped when ``exclude_ignore_regions`` is True).

    If ``mask`` is absent, every patch is considered clean.

    Parameters
    ----------
    hdf5_path
        Input HDF5 file.
    output_dir
        Directory to write ``.png`` patch files (created if missing).
    patch_size
        Side length of each square patch in pixels.
    stride
        Step between patch origins; defaults to ``patch_size`` (non-overlapping grid).
    max_patches
        Stop after saving this many patches (scan order: top-to-bottom, left-to-right).
    exclude_ignore_regions
        If True, drop any patch that overlaps a non-zero ``ignore_mask`` pixel.
    filename_prefix
        Prepended to each filename; default is the HDF5 stem.

    Returns
    -------
    list[Path]
        Paths of written patch images.
    """
    hdf5_path = Path(hdf5_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stride is None:
        stride = patch_size

    pil_image, gt_mask, ignore_mask = load_hdf5_input(hdf5_path)
    rgb = np.asarray(pil_image.convert("RGB"))
    h, w = rgb.shape[0], rgb.shape[1]

    if gt_mask is not None:
        anomaly = (align_mask_to_image(gt_mask, (h, w)) > 0).astype(np.uint8)
    else:
        anomaly = np.zeros((h, w), dtype=np.uint8)

    if exclude_ignore_regions and ignore_mask is not None:
        ignore = (align_mask_to_image(ignore_mask, (h, w)) != 0).astype(np.uint8)
    else:
        ignore = np.zeros((h, w), dtype=np.uint8)

    prefix = filename_prefix if filename_prefix is not None else hdf5_path.stem
    saved: list[Path] = []

    for y0 in range(0, h - patch_size + 1, stride):
        for x0 in range(0, w - patch_size + 1, stride):
            if anomaly[y0 : y0 + patch_size, x0 : x0 + patch_size].any():
                continue
            if ignore[y0 : y0 + patch_size, x0 : x0 + patch_size].any():
                continue

            patch = rgb[y0 : y0 + patch_size, x0 : x0 + patch_size]
            out_name = f"{prefix}_patch_y{y0}_x{x0}.png"
            out_path = output_dir / out_name
            Image.fromarray(patch, mode="RGB").save(out_path)
            saved.append(out_path)

            if max_patches is not None and len(saved) >= max_patches:
                return saved

    return saved
