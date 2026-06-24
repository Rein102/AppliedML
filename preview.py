import os
import h5py
import matplotlib.pyplot as plt

ROOT = ""

files = []

for dirpath, _, filenames in os.walk(ROOT):
    for f in filenames:
        if f.endswith(".hdf5"):
            files.append(os.path.join(dirpath, f))

files = sorted(files)

print("files:", len(files))

for path in files:

    with h5py.File(path, "r") as f:
        img = f["img"][:]

        mask = None
        if "mask" in f:
            mask = f["mask"][:]

    name = os.path.basename(path)

    print(name)
    print("shape:", img.shape)

    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.title(name)
    plt.axis("off")
    plt.show()

    if mask is not None:
        plt.figure(figsize=(12, 12))
        plt.imshow(mask, cmap="gray")
        plt.title(name + " MASK")
        plt.axis("off")
        plt.show()