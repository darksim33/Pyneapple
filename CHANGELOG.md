# Changelog

> **TL;DR** — Full version history for Pyneapple. Each entry covers Added, Changed, Fixed, and Removed changes per release. Versions v1.6.0–v1.6.2 are intermediate releases between the v1.5.6 and v1.7.0 tags. Pre-releases (v0.6.0–v0.7.3) are noted as such.

---

## [Unreleased]

### Changed

- Minimum supported Python version lowered from 3.12 to **3.9**
- `tomli>=2.0` added as a conditional dependency (`python_version < "3.11"`) to backport the stdlib `tomllib` module
- Ruff `target-version` updated to `py39`; Black `target-version` extended to cover py39–py312

### Fixed

- `import tomllib` in `src/pyneapple/io/toml.py` replaced with a `try/except ImportError` shim that falls back to `tomli` on Python < 3.11
- `isinstance(value, h5py.Group | h5py.Dataset)` in `src/pyneapple/io/hdf5.py` replaced with the tuple form `isinstance(value, (h5py.Group, h5py.Dataset))` — runtime `|` union in `isinstance()` requires Python 3.10+
- Missing `from __future__ import annotations` added to `src/pyneapple/fitters/ideal.py` — without it, `X | Y` union annotations in function signatures were evaluated at class definition time and raised `TypeError` on Python 3.9

---

## [v2.0.0-beta.2] — 2026-04-22

### Fixed

- `SegmentedFitter` was missing from `pyneapple.fitters.__init__` — it is now properly exported and available as `from pyneapple.fitters import SegmentedFitter`

---

## [v2.0.0-beta.1] — 2026-04-01

> ⚠️ **Major breaking changes** — This release restructures the entire codebase to follow scikit-learn conventions. See [release_draft.md](./release_draft.md) for comprehensive migration guide.

### Added

- **scikit-learn Style Architecture**: Modular structure with `Models`, `Solvers`, and `Fitters` following sklearn estimator patterns
- **Models**: Stateless, pure math forward model functions (`forward`, `jacobian`, `residual`) — `MonoExpModel`, `BiExpModel`, `TriExpModel`, `NNLSModel`
- **Solvers**: Curve fitting backends — `CurveFitSolver`, `NNLSSolver`, `ConstrainedCurveFitSolver`
- **Fitters**: High-level fitting orchestrators — `PixelWiseFitter`, `SegmentationWiseFitter`, `IDEALFitter`, `SegmentedFitter`
- **Registry-based plugin discovery** via entry points for models, solvers, and fitters
- **`pyneapple.io` module**: New I/O utilities for NIfTI, b-values, TOML configs, Excel, and HDF5
- **`pyneapple.utility` module**: New utilities for spectrum processing, plotting, and validation
- **New CLI tools**: `pyneapple pixelwise`, `pyneapple segmented`, `pyneapple ideal`, `pyneapple segmentationwise`
- **Optional dependencies** support: `pip install pyneapple[excel,plotting,export]`
- **New TOML config format** with `[Fitting]` section replacing old `[General]` with `Class` field
- **SegmentedFitter** for two-step model fitting workflow
- **NIfTI export** with `reconstruct_maps` and `save_spectrum_to_nifti`
- **Excel export** with `save_params_to_excel` and `save_spectrum_to_excel`
- **HDF5 export** with gzip compression

### Changed

- **TOML config format redesigned**: Removed `Class` field, restructured under `[Fitting]` section
- **CLI interface redesign**: `pyneapple pixelwise/segmented/ideal/segmentationwise` replaces `pyneapple-fitdata`
- **Model naming unified**: `BiExpFitModel` → `BiExpModel`, `MonoExpFitModel` → `MonoExpModel`, etc.
- **Logging API**: `configure_logging()` from `pyneapple` replaces `from pyneapple.utils import logger`
- Moved from `poetry` to `uv` for package management

### Removed

- **`pyneapple/parameters/` module entirely**: `Parameters`, `IVIMParams`, `IVIMSegmentedParams`, `NNLSParams`, `NNLSCVParams`, `IDEALParams`, `BaseBoundaryDict`
- **`pyneapple/results/` module entirely**: `Results`, `IVIMResults`, `IVIMSegmentedResults`, `NNLSResults`, `ResultDict`
- **`pyneapple/fitting/` module entirely**: `FitData` and all fitting functions
- **`pyneapple/utils/` module entirely**: `logger`, `plotting`, `processing`
- **`radimgarray` submodule**: `RadImgArray`, `SegImgArray` no longer available
- **`pygpufit` module**: Moved to separate package
- **JSON/TOML parameter files**: All files in `examples/parameters/` replaced with new format
- **Example scripts**: `ivim_fitdata_script.py`, `nnls_fitdata_script.py`
- **Old fitting modes**: `fit_pixel_wise()`, `fit_segmentation_wise()`, `fit_ivim_segmented()`, `fit_ideal()`

### Fixed

- Various import path corrections and cleanup

---

## [v1.7.0] — 2026-01-19

### Added

- HDF5 data storage and export (3D/4D support, dict import, `sparse` library for N-D arrays) ([#198](https://github.com/darksim33/Pyneapple/pull/198), [#199](https://github.com/darksim33/Pyneapple/pull/199))
- NIfTI helper functions and refactored results saving for NIfTI format
- Dedicated IO module consolidating JSON, TOML, NIfTI, and HDF5 operations
- `fit_opt` parameter to track fitting type (pixelwise / segmentation / segmented / ideal)
- NNLS peak merging and cutoff handling
- Comprehensive test suites for HDF5, NIfTI, and IO integration

### Changed

- Switched from `scipy.sparse` to `sparse` library for N-D array support
- GPU fitter model retrieval and constraint handling improved ([#197](https://github.com/darksim33/Pyneapple/pull/197))
- `DataExport` documentation expanded

### Fixed

- `fit_opt` defaulting to empty string
- GPU fitter constraints and import errors
- NNLS model params not passed properly
- Shape calculation issues
- Missing logger output

---

## [v1.6.2] *(intermediate — not a published tag)*

### Added

- `fit_t1_steam` option for T1 STEAM correction ([#193](https://github.com/darksim33/Pyneapple/pull/193))
- Example TOML files for BiExp IVIM and NNLS fitting
- Fit example scripts and extended example descriptions
- T1 fit check during results export

### Changed

- IVIM results refactored: unified S0 and fraction extraction method
- `dimstep` now always returns `np.ndarray`
- Class descriptions and type hints updated throughout

### Fixed

- Segmented test issues after IVIM results rework
- Additional small-fraction extraction edge case
- Parameter class invoke error
- JSON-to-`params_file` conversion issues

---

## [v1.6.1] *(intermediate — not a published tag)*

### Added

- TOML parameter file import and export ([#168](https://github.com/darksim33/Pyneapple/pull/168))
- TOML test suite and import examples
- TOML documentation
- Dependencies: `tomli`, `tomli-w`, `tomlkit`

### Changed

- `params.json` renamed to `params.file` to support multiple formats
- Backward compatible with existing JSON parameter files

---

## [v1.6.0] *(intermediate — not a published tag)*

### Added

- Custom loguru-based logger ([#167](https://github.com/darksim33/Pyneapple/pull/167))
- Configurable log levels; file output support
- Logger tests; log level set to `ERROR` for pytest runs
- Linux GPUFit library (Ubuntu 22.04 LTS)
- Model class restructuring: base fit model classes, explicit mono/bi/tri-exponential models, GPU model constructor ([#171](https://github.com/darksim33/Pyneapple/pull/171))
- Dynamic dict-based parameter handling replacing static file loading ([#184](https://github.com/darksim33/Pyneapple/pull/184))

### Changed

- Replaced `print` statements throughout codebase with loguru calls
- `reduced` renamed to `fit_reduced`; `fit_s0` and `fit_t1` properly propagated
- Model wrapper pattern replaced with direct model classes
- Moved parameter file fixtures from `fitting/` to `parameters/`
- `*_comment*` attributes renamed to `*description*`
- `tox.ini` moved into `pyproject.toml`

### Fixed

- `LOG_LEVEL` variable naming
- Removed unwanted global variables and file output side effects
- Missing segmented S0 value
- `kwargs` overwriting boolean settings

---

## [v1.5.6] — 2025-03-21

### Added

- Full GPUfit integration for CUDA GPU fitting
- Explicit mono-, bi-, and tri-exponential models (replaces multi-exponential wrapper)
- New result classes for structured IVIM and NNLS output
- Individual pixel-by-pixel boundary support (`btype`) ([#197](https://github.com/darksim33/Pyneapple/pull/197))
- pytest-mock and pytest-cov; fitting, boundary, and GPU fitter test suites
- Python 3.13 support

### Changed

- Model wrapper deprecated in favor of explicit model classes
- Parameter files restructured; files from v1.3.1 or older are no longer supported
- Segmented fitting reworked with two sub-parameter classes
- Boundary system reworked to altered dict structure with order handling

### Fixed

- MonoExp equation order (`exp(D1) * S0`)
- `fit_s0` addition to `params2`
- `D1` position in first segmented fit pass
- NNLS results boolean parsing and variable array length

---

## [v1.5.5] — 2025-03-20

### Fixed

- Removed IVIM parameter reorder array — all models now use the same parameter set

---

## [v1.5.4] — 2025-03-12

### Fixed

- S0 and fraction extraction issues for IVIM fitting

---

## [v1.5.3] — 2025-03-07

### Fixed

- Typos, broken conditionals, and `fit_model` selection issues

---

## [v1.5.1] — 2024-12-11

### Changed

- IDEAL fitting moved to a separate branch; IDEAL files removed from `main` until stable ([#156](https://github.com/darksim33/Pyneapple/issues/156))

---

## [v1.5.0] — 2024-12-11

### Added

- Full GPU integration for all supported model types
- `kwargs` passing for GPU fitter tuning

### Changed

- Fitting models reworked to share a unified interface across CPU and GPU backends ([#158](https://github.com/darksim33/Pyneapple/pull/158))

---

## [v1.4.0] — 2024-12-05

### Added

- GPU fitting via `pygpufit` for CUDA-capable GPUs ([#150](https://github.com/darksim33/Pyneapple/pull/150))

---

## [v1.3.1] — 2024-11-09

### Added

- `RadImgArray` implementation for image array handling ([#145](https://github.com/darksim33/Pyneapple/pull/145))

### Changed

- Reconstruction split into UI and fitting layers ([#141](https://github.com/darksim33/Pyneapple/pull/141))
- Results classes reworked ([#146](https://github.com/darksim33/Pyneapple/pull/146))

### Removed

- UI components extracted from the fitting library

---

## [v1.1.1] — 2024-10-01

### Added

- Load b-values option via context menu ([#131](https://github.com/darksim33/Pyneapple/pull/131))
- IVIM fixed parameters support ([#136](https://github.com/darksim33/Pyneapple/pull/136))
- Python 3.9 support ([#138](https://github.com/darksim33/Pyneapple/pull/138))

### Changed

- Fit results converted to custom dict types ([#121](https://github.com/darksim33/Pyneapple/pull/121))
- `fit_data` restructured for clarity ([#133](https://github.com/darksim33/Pyneapple/pull/133))

### Fixed

- Save results bug ([#127](https://github.com/darksim33/Pyneapple/pull/127))

---

## [v1.0.0] — 2024-04-29

First public release.

### Added

- Pixel-wise and segmentation-wise IVIM fitting (mono-, bi-, tri-exponential models)
- NNLS spectral fitting
- JSON parameter file support
- Core test suite
- Basic UI (subsequently removed in v1.3.1)

### Notes

- IDEAL fitting shipped as work-in-progress

---

## [v0.7.3] — 2024-04-17 *(pre-release)*

### Added

- New package structure with icons and cleaned directory layout
- NNLS and NNLS-regularised parameter unification ([#88](https://github.com/darksim33/Pyneapple/pull/88))
- File dialog memory (`last_dir` property) ([#96](https://github.com/darksim33/Pyneapple/pull/96))

### Fixed

- `reg = 0` fitting edge case ([#59](https://github.com/darksim33/Pyneapple/pull/59))
- Fit area selection ([#64](https://github.com/darksim33/Pyneapple/pull/64))
- Zero-padding ([#89](https://github.com/darksim33/Pyneapple/pull/89))
- IDEAL fitting fixes ([#85](https://github.com/darksim33/Pyneapple/pull/85))
- Various UI and import cleanup ([#67](https://github.com/darksim33/Pyneapple/pull/67), [#97](https://github.com/darksim33/Pyneapple/pull/97))

---

## [v0.7.0] — 2024-02-15 *(pre-release)*

### Added

- Initial IDEAL fitting support

---

## [v0.6.1] — 2023-12-14 *(pre-release)*

Minor patch. No release notes available.

---

## [v0.6.0] — 2023-12-13 *(pre-release)*

Initial pre-release. No release notes available.
