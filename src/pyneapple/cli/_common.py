"""Shared pipeline utilities used by all Pyneapple CLI entry points."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from pyneapple.fitters.segmentationwise import SegmentationWiseFitter

from ..io import (
    load_dwi_nifti,
    load_bvalues,
    save_parameter_map,
    reconstruct_maps,
    reconstruct_segmentation_maps,
)
from ..io.toml import load_config

if TYPE_CHECKING:
    import argparse


# ---------------------------------------------------------------------------
# Shared argument builder
# ---------------------------------------------------------------------------


def add_shared_args(
    parser: "argparse.ArgumentParser",
    *,
    seg_required: bool = False,
) -> None:
    """Register all arguments that are common to every Pyneapple CLI tool.

    Args:
        parser: The parser to add arguments to.
        seg_required: When ``True``, ``--seg`` is a required argument (needed
            for segmentation-wise and IDEAL fitting).
    """
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
        required=seg_required,
        default=None,
        type=Path,
        metavar="PATH",
        help=(
            "Segmentation / ROI mask NIfTI file (.nii / .nii.gz). "
            "Non-zero voxels are fitted; all others are skipped. "
            "Must match the spatial shape of the DWI image."
            + (" Required for this fitting mode." if seg_required else " Optional.")
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


# ---------------------------------------------------------------------------
# Shared fitting pipeline
# ---------------------------------------------------------------------------


def run_pipeline(args: "argparse.Namespace") -> int:
    """Execute the end-to-end fitting pipeline for any Pyneapple CLI tool.

    Runs the following steps in order:

    1. Load DWI image, b-values, and optional segmentation mask.
    2. Load TOML config and build the fitter.
    3. Load optional per-pixel fixed parameter maps.
    4. Run fitting.
    5. Reconstruct spatial parameter maps.
    6. Save one NIfTI per parameter.

    Args:
        args: Parsed arguments. Expected attributes: ``image``, ``bval``,
            ``config``, ``seg`` (``None`` or ``Path``), ``output`` (``None``
            or ``Path``), ``verbose`` (bool), ``fixed`` (list or ``None``).

    Returns:
        int: Exit code: ``0`` on success, ``1`` on user error, ``2`` on
            missing file.
    """
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
                name, nifti_path_str = spec.split(":", 1)
                nifti_path = Path(nifti_path_str)
                logger.info(f"Loading fixed param map: {name} ← {nifti_path}")
                map_data, _ = load_dwi_nifti(str(nifti_path))
                map_3d = map_data[..., 0]
                if map_3d.shape != spatial_shape:
                    raise ValueError(
                        f"Fixed param map '{name}' has spatial shape "
                        f"{map_3d.shape}, expected {spatial_shape}."
                    )
                fixed_param_maps[name] = map_3d

        # ------------------------------------------------------------------
        # 4. Fit
        # ------------------------------------------------------------------
        logger.info("Starting fitting …")
        fitter.fit(
            bvalues,
            image,
            segmentation=segmentation,
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
            param_maps = {
                k: np.atleast_1d(np.asarray(v, dtype=np.float32))
                for k, v in fitter.fitted_params_.items()
            }
        elif (
            isinstance(fitter, SegmentationWiseFitter)
            and fitter.pixel_to_segment is not None
        ):
            param_maps = reconstruct_segmentation_maps(
                fitter.fitted_params_,
                fitter.pixel_to_segment,
                len(fitter.segment_labels),  # type: ignore handled by SegmentationWiseFitter
                spatial_shape,
            )
        else:
            # per-pixel fitter (pixelwise, IDEAL ...) with valid pixel_indices
            param_maps = reconstruct_maps(
                fitter.fitted_params_, fitter.pixel_indices, spatial_shape
            )

        # ------------------------------------------------------------------
        # 6. Save one NIfTI per parameter
        # ------------------------------------------------------------------
        output_dir = Path(args.output) if args.output else args.image.parent
        output_dir.mkdir(parents=True, exist_ok=True)

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
