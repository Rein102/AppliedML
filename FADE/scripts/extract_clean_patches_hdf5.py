#!/usr/bin/env python3
"""CLI: extract clean patches from an HDF5 file into a folder."""

from pathlib import Path

import click

from utils.hdf5_patches import extract_clean_patches_from_hdf5


@click.command()
@click.argument("hdf5_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option("--patch-size", type=int, default=128, show_default=True)
@click.option("--stride", type=int, default=None, help="Defaults to patch-size (grid, no overlap).")
@click.option("--max-patches", type=int, default=None, help="Cap number of saved patches.")
@click.option(
    "--allow-ignore-mask-overlap",
    is_flag=True,
    default=False,
    help="If set, keep patches that overlap ignore_mask (default: skip those patches).",
)
@click.option("--prefix", type=str, default=None, help="Filename prefix (default: HDF5 stem).")
def main(
    hdf5_path: Path,
    output_dir: Path,
    patch_size: int,
    stride: int | None,
    max_patches: int | None,
    allow_ignore_mask_overlap: bool,
    prefix: str | None,
):
    paths = extract_clean_patches_from_hdf5(
        hdf5_path,
        output_dir,
        patch_size=patch_size,
        stride=stride,
        max_patches=max_patches,
        exclude_ignore_regions=not allow_ignore_mask_overlap,
        filename_prefix=prefix,
    )
    click.echo(f"Saved {len(paths)} patch(es) under {output_dir}")


if __name__ == "__main__":
    main()
