import sys
import h5py
import numpy as np
from PIL import Image
from pathlib import Path

hdf5_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

with h5py.File(hdf5_path, "r") as f:
    img = f["img"][:]

if img.ndim == 2:
    img = np.stack([img, img, img], axis=-1)

if img.dtype != np.uint8:
    img = (img / img.max() * 255).astype(np.uint8)

Image.fromarray(img).save(out_path)
print(f"Saved {img.shape} image to {out_path}")
