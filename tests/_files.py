"""
Define different files needed for testing to use in pytest fixtures.
"""

import pytest
import json
import tempfile
import numpy as np
from pathlib import Path

from radimgarray import RadImgArray, SegImgArray


# --- Parameters ---


class ParameterTools:

    @staticmethod
    def deploy_temp_file(file: Path | str):
        """Yield file and unlink afterwards."""
        if isinstance(file, str):
            file = Path(file)
        yield file
        if file.exists():
            file.unlink()

    @staticmethod
    def dict_to_json(data: dict, _dir: str) -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=_dir
        ) as f:
            json.dump(data, f, indent=4)
            temp_file = Path(f.name)
            return temp_file

    @staticmethod
    def change_keys(params: dict, changed_params: dict) -> dict:
        """
        Change keys in the params dictionary based on the changed_params dictionary.
        """
        for key, value in changed_params.items():
            if "." in key:
                # Handle nested dictionary access
                category, subkey = key.split(".", 1)
                if category in params and subkey in params[category]:
                    params[category][subkey] = value
                else:
                    raise KeyError(f"Nested key '{key}' not found in parameters.")
                # Make sure T1 boundaries are passed if fit_t1 is set
                if (
                    subkey == "fit_t1"
                    and value
                    and not params["Boundaries"].get("T", False)
                ):
                    params["Boundaries"]["T"] = {"1": [2000, 10, 10000]}
            elif key in params:
                # Handle top-level keys
                params[key] = value
            else:
                raise KeyError(f"Key '{key}' not found in parameters.")
        return params

    @staticmethod
    def get_basic_parameters() -> dict:
        return {
            "General": {
                "Class": "",
                "description": "",
                "fit_type": "",
                "max_iter": 250,
                "n_pools": 4,
                "fit_tolerance": 1e-6,
                "b_values": [
                    0,
                    10,
                    20,
                    30,
                    40,
                    50,
                    70,
                    100,
                    150,
                    200,
                    250,
                    350,
                    450,
                    550,
                    650,
                    750,
                ],
            },
            "Model": {},
            "Boundaries": {},
        }

    # --- IVIM Parameters ---
    @staticmethod
    def get_basic_ivim_parameters() -> dict:
        params = ParameterTools.get_basic_parameters()
        params["General"]["Class"] = "IVIMParams"
        params["General"]["fit_type"] = "single"
        params["Model"]["fit_reduced"] = False
        params["Model"]["fit_S0"] = False
        params["Model"]["fit_t1"] = False
        params["Model"]["mixing_time"] = None
        params["Boundaries"]["f"] = {}
        params["Boundaries"]["D"] = {}
        return params

    @staticmethod
    def get_basic_ivim_mono():
        params = ParameterTools.get_basic_ivim_parameters()
        params["Model"]["model"] = "monoexp"
        params["Boundaries"]["f"]["1"] = [210, 10, 10000]
        params["Boundaries"]["D"]["1"] = [0.001, 0.0007, 0.3]
        return params

    @staticmethod
    def get_basic_ivim_biexp():
        params = ParameterTools.get_basic_ivim_parameters()
        params["Model"]["model"] = "biexp"
        params["Boundaries"]["f"]["1"] = [85, 10, 500]
        params["Boundaries"]["f"]["2"] = [20, 1, 100]
        params["Boundaries"]["D"]["1"] = [0.001, 0.0007, 0.05]
        params["Boundaries"]["D"]["2"] = [0.02, 0.003, 0.3]
        return params

    @staticmethod
    def get_basic_ivim_triexp():
        params = ParameterTools.get_basic_ivim_parameters()
        params["Model"]["model"] = "triexp"
        params["Boundaries"]["f"]["1"] = [85, 10, 500]
        params["Boundaries"]["f"]["2"] = [20, 1, 100]
        params["Boundaries"]["f"]["3"] = [20, 1, 100]
        params["Boundaries"]["D"]["1"] = [0.001, 0.0007, 0.05]
        params["Boundaries"]["D"]["2"] = [0.02, 0.003, 0.3]
        params["Boundaries"]["D"]["3"] = [0.01, 0.003, 0.3]
        return params

    @staticmethod
    def add_basic_segmented(params: dict) -> dict:
        params["General"]["Class"] = "IVIMSegmentedParams"
        params["General"]["fixed_component"] = "D_1"
        params["General"]["fixed_t1"] = False
        params["General"]["reduced_bvalues"] = False
        return params

    @staticmethod
    def add_ideal(params: dict) -> dict:
        params["General"]["Class"] = "IDEALParams"
        params["General"]["dim_steps"] = [
            [176, 176],
            [156, 156],
            [128, 128],
            [96, 96],
            [64, 64],
            [32, 32],
            [16, 16],
            [8, 8],
            [4, 4],
            [2, 2],
            [1, 1],
        ]
        params["General"]["step_tol"] = [0.2, 0.2, 0.1, 0.2]
        params["General"]["segmentation_threshold"] = 0.025
        return params

    # --- NNLS Parameters ---

    @staticmethod
    def add_nnls(params: dict) -> dict:
        params["General"]["Class"] = "NNLSParams"
        params["General"]["max_iter"] = 600
        params["Model"]["model"] = "nnls"
        params["Model"]["mu"] = 0.02
        params["Model"]["reg_order"] = 0
        params["Boundaries"]["d_range"] = [0.0007, 0.3]
        params["Boundaries"]["n_bins"] = 350
        return params

    @staticmethod
    def add_nnls_cv(params: dict) -> dict:
        params["General"]["Class"] = "NNLSCVParams"
        params["Model"]["tol"] = 1e-4
        return params


# --- Fixtures for Files ---


@pytest.fixture
def out_json(temp_dir):
    """Fixture to create a temporary JSON file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=temp_dir
    ) as f:
        yield Path(f.name)
        f.close()
        if Path(f.name).exists():
            Path(f.name).unlink()


@pytest.fixture
def out_nii(temp_dir):
    """Fixture to create a temporary NIfTI file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".nii.gz", delete=False, dir=temp_dir
    ) as f:
        yield Path(f.name)
        f.close()
        if Path(f.name).exists():
            Path(f.name).unlink()


@pytest.fixture
def out_excel(temp_dir):
    """Fixture to create a temporary Excel file."""
    file = temp_dir / "test.xlsx"
    yield file
    if file.exists():
        file.unlink()


# --- Parameter Files ---


@pytest.fixture
def ivim_mono_params_file(temp_dir):
    params = ParameterTools.get_basic_ivim_mono()
    yield from ParameterTools.deploy_temp_file(
        ParameterTools.dict_to_json(params, temp_dir)
    )


# --- IVIM Biexp Parameter Files ---


@pytest.fixture
def ivim_bi_params_file(temp_dir):
    params = ParameterTools.get_basic_ivim_biexp()
    yield from ParameterTools.deploy_temp_file(
        ParameterTools.dict_to_json(params, temp_dir)
    )


@pytest.fixture
def ivim_bi_t1_params_file(temp_dir):
    params = ParameterTools.get_basic_ivim_biexp()
    params = ParameterTools.change_keys(
        params, {"Model.fit_t1": True, "Model.mixing_time": 20}
    )
    yield from ParameterTools.deploy_temp_file(
        ParameterTools.dict_to_json(params, temp_dir)
    )


@pytest.fixture
def ivim_bi_segmented_params_file(temp_dir):
    params = ParameterTools.get_basic_ivim_biexp()
    params = ParameterTools.add_basic_segmented(params)
    yield from ParameterTools.deploy_temp_file(
        ParameterTools.dict_to_json(params, temp_dir)
    )


@pytest.fixture
def ideal_params_file(temp_dir):
    params = ParameterTools.get_basic_ivim_biexp()
    params = ParameterTools.add_ideal(params)
    yield from ParameterTools.deploy_temp_file(
        ParameterTools.dict_to_json(params, temp_dir)
    )


# --- IVIM Triexp Parameter Files ---


@pytest.fixture
def ivim_tri_params_file(temp_dir):
    params = ParameterTools.get_basic_ivim_triexp()
    yield from ParameterTools.deploy_temp_file(
        ParameterTools.dict_to_json(params, temp_dir)
    )


@pytest.fixture
def ivim_tri_t1_params_file(temp_dir):
    params = ParameterTools.get_basic_ivim_triexp()
    params = ParameterTools.change_keys(
        params, {"Model.fit_t1": True, "Model.mixing_time": 20}
    )
    yield from ParameterTools.deploy_temp_file(
        ParameterTools.dict_to_json(params, temp_dir)
    )


@pytest.fixture
def ivim_tri_t1_no_mixing_params_file(temp_dir):
    params = ParameterTools.get_basic_ivim_triexp()
    params = ParameterTools.change_keys(params, {"Model.fit_t1": True})
    yield from ParameterTools.deploy_temp_file(
        ParameterTools.dict_to_json(params, temp_dir)
    )


@pytest.fixture
def ivim_tri_segmented_params_file(temp_dir):
    params = ParameterTools.get_basic_ivim_triexp()
    params = ParameterTools.add_basic_segmented(params)
    yield from ParameterTools.deploy_temp_file(
        ParameterTools.dict_to_json(params, temp_dir)
    )


@pytest.fixture
def ivim_tri_t1_segmented_params_file(temp_dir):
    params = ParameterTools.get_basic_ivim_triexp()
    params = ParameterTools.add_basic_segmented(params)
    params = ParameterTools.change_keys(
        params, {"Model.fit_t1": True, "Model.mixing_time": 20}
    )
    yield from ParameterTools.deploy_temp_file(
        ParameterTools.dict_to_json(params, temp_dir)
    )


# --- NNLS Parameter Files ---


@pytest.fixture
def nnls_params_file(temp_dir):
    params = ParameterTools.get_basic_parameters()
    params = ParameterTools.add_nnls(params)
    yield from ParameterTools.deploy_temp_file(
        ParameterTools.dict_to_json(params, temp_dir)
    )


@pytest.fixture
def nnls_cv_params_file(temp_dir):
    params = ParameterTools.get_basic_parameters()
    params = ParameterTools.add_nnls(params)
    params = ParameterTools.add_nnls_cv(params)
    yield from ParameterTools.deploy_temp_file(
        ParameterTools.dict_to_json(params, temp_dir)
    )
