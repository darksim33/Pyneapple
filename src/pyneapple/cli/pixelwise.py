"""CLI entry point: pixelwise diffusion MRI fitting.

Usage
-----
::

    pyneapple-pixelwise \\
        --image  dwi.nii.gz \\
        --bval   dwi.bval \\
        --config config.toml \\
        [--seg   mask.nii.gz] \\
        [--output ./results] \\
        [--verbose]

Outputs one NIfTI parameter map per fitted parameter, named
``<image_stem>_<param>.nii.gz`` in the chosen output directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from loguru import logger

from ..io import load_dwi_nifti, load_bvalues, save_parameter_map
from ..io.toml import load_config


# ---------------------------------------------------------------------------
# Spatial map reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_maps(
    fitted_params: dict[str, np.ndarray],
    pixel_indices: list[tuple[int, ...]],
    spatial_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Map 1-D per-pixel arrays back to their spatial positions.

    Parameters
    ----------
    fitted_params : dict[str, ndarray]
        Dictionary of parameter name → 1-D array of shape ``(n_pixels,)``.
    pixel_indices : list[tuple[int, ...]]
        Spatial index for each pixel in ``fitted_params`` values, as returned
        by :attr:`BaseFitter.pixel_indices`.
    spatial_shape : tuple[int, ...]
        Spatial shape of the output volume (e.g. ``(X, Y, Z)``).

    Returns
    -------
    dict[str, ndarray]
        Dictionary of parameter name → 3-D array of shape ``spatial_shape``
        (zero-filled where no pixel was fitted).
    """
    maps: dict[str, np.ndarray] = {}
    idx = tuple(zip(*pixel_indices))  # unzip to per-dimension index arrays

    for param, values in fitted_params.items():
        values = values.astype(np.float32)
        # 1-D per-pixel → 3-D spatial volume; 2-D per-pixel → 4-D spatial volume
        extra_dims = values.shape[1:] if values.ndim > 1 else ()
        vol = np.zeros(spatial_shape + extra_dims, dtype=np.float32)
        vol[idx] = values
        maps[param] = vol

    return maps


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyneapple-pixelwise",
        description="Pixelwise diffusion MRI fitting with Pyneapple.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--image",
        "-i",
        required=True,
        type=Path,
        metavar="PATH",
        help="4-D DWI NIfTI image (.nii / .nii.gz).",
    )
    parser.add_argument(
        "--bval",
        "-b",
        required=True,
        type=Path,
        metavar="PATH",
        help="B-value file (.bval / .txt), one value per line or space-separated.",
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=Path,
        metavar="PATH",
        help="TOML fitting configuration file.",
    )
    parser.add_argument(
        "--seg",
        "-s",
        default=None,
        type=Path,
        metavar="PATH",
        help=(
            "Segmentation / ROI mask NIfTI file (.nii / .nii.gz). "
            "Non-zero voxels are fitted; all others are skipped. "
            "Must match the spatial shape of the DWI image. Optional."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        type=Path,
        metavar="DIR",
        help=(
            "Output directory for parameter maps. "
            "Defaults to the directory containing the DWI image."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--fixed",
        "-f",
        action="append",
        default=None,
        metavar="NAME:PATH",
        help=(
            "Fix a model parameter to a per-pixel NIfTI map. "
            "Format: NAME:PATH where NAME is the parameter name and "
            "PATH is a 3-D NIfTI file (.nii / .nii.gz) whose spatial shape "
            "matches the DWI image. May be repeated for multiple parameters."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code: 0 on success, non-zero on failure.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure logging level
    logger.remove()
    level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=level, colorize=True)

    try:
        # ------------------------------------------------------------------
        # 1. Load inputs
        # ------------------------------------------------------------------
        logger.info(f"Loading DWI image:  {args.image}")
        image, ref_nifti = load_dwi_nifti(str(args.image))

        logger.info(f"Loading b-values:   {args.bval}")
        bvalues = load_bvalues(str(args.bval))
        logger.debug(f"  b-values: {bvalues}")

        segmentation: np.ndarray | None = None
        if args.seg is not None:
            logger.info(f"Loading segmentation: {args.seg}")
            # load_dwi_nifti handles validation and normalises to 4-D;
            # take the first volume to obtain a 3-D integer mask.
            seg_data, _ = load_dwi_nifti(str(args.seg))
            segmentation = seg_data[..., 0].astype(np.int32)
            n_roi = int(np.count_nonzero(segmentation))
            logger.info(f"  Segmentation: {n_roi} non-zero voxels")

        logger.debug(
            f"Image shape: {image.shape} | b-values: {len(bvalues)} | "
            f"Segmentation: {segmentation.shape if segmentation is not None else 'None (full image)'}"
        )

        # ------------------------------------------------------------------
        # 2. Load config and build fitter
        # ------------------------------------------------------------------
        logger.info(f"Loading config:     {args.config}")
        config = load_config(args.config)
        fitter = config.build_fitter()

        # ------------------------------------------------------------------
        # 3. Load per-pixel fixed parameter maps (if any)
        # ------------------------------------------------------------------
        fixed_param_maps: dict[str, np.ndarray] | None = None
        if args.fixed:
            fixed_param_maps = {}
            spatial_shape = image.shape[:3]
            for spec in args.fixed:
                if ":" not in spec:
                    raise ValueError(
                        f"Invalid --fixed format: {spec!r}. "
                        "Expected NAME:PATH (e.g. T1:/path/to/t1_map.nii.gz)."
                    )
                name, nifti_path = spec.split(":", 1)
                nifti_path = Path(nifti_path)
                logger.info(f"Loading fixed param map: {name} \u2190 {nifti_path}")
                map_data, _ = load_dwi_nifti(str(nifti_path))
                map_3d = map_data[..., 0]  # first volume → 3-D
                if map_3d.shape != spatial_shape:
                    raise ValueError(
                        f"Fixed param map '{name}' has spatial shape "
                        f"{map_3d.shape}, expected {spatial_shape} "
                        f"(matching the DWI image)."
                    )
                fixed_param_maps[name] = map_3d

        # ------------------------------------------------------------------
        # 4. Fit
        # ------------------------------------------------------------------
        logger.info("Starting pixelwise fitting \u2026")
        fitter.fit(
            bvalues, image, segmentation=segmentation,
            fixed_param_maps=fixed_param_maps,
        )
        logger.info("Fitting complete.")

        # ------------------------------------------------------------------
        # 5. Reconstruct spatial parameter maps
        # ------------------------------------------------------------------
        spatial_shape = image.shape[:3]

        if fitter.pixel_indices is None:
            logger.warning(
                "fitter.pixel_indices is None — cannot reconstruct spatial maps. "
                "Saving raw 1-D parameter arrays instead."
            )
            param_maps = fitter.fitted_params_
        else:
            param_maps = _reconstruct_maps(
                fitter.fitted_params_, fitter.pixel_indices, spatial_shape
            )

        # ------------------------------------------------------------------
        # 6. Save one NIfTI per parameter
        # ------------------------------------------------------------------
        output_dir = Path(args.output) if args.output else args.image.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use the image stem, stripping any .nii / .nii.gz suffix
        stem = args.image.name
        for suffix in (".nii.gz", ".nii"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break

        saved: list[Path] = []
        for param_name, vol in param_maps.items():
            out_path = output_dir / f"{stem}_{param_name}.nii.gz"
            save_parameter_map(
                {param_name: vol},
                str(out_path),
                ref_nifti,
                param_name=param_name,
            )
            saved.append(out_path)
            logger.info(f"  Saved {param_name} → {out_path}")

        logger.info(f"Done. {len(saved)} parameter map(s) written to '{output_dir}'.")
        return 0

    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 2
    except (ValueError, KeyError) as exc:
        logger.error(str(exc))
        return 1
    except Exception as exc:  # pragma: no cover
        logger.exception(f"Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
