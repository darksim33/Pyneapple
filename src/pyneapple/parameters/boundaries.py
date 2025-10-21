"""Module for handling boundaries for fitting parameters."""

from __future__ import annotations

import numpy as np

from ..utils.logger import logger


class BaseBoundaryDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_axis_limits(self) -> tuple:
        """Get Limits for axis in parameter values."""
        return 0.0001, 1

    @property
    def parameter_names(self) -> list | dict:
        """Get list of parameter names."""
        names = []
        for key, value in self.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    names.append(key + "_" + sub_key)
            else:
                names.append(key)
        return names


class IVIMBoundaryDict(BaseBoundaryDict):
    """Dictionary class for handling IVIM boundaries."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def btype(self) -> str:
        """Get boundary type.
        Types: "general", "individual"
        """
        _btype = "general"
        for key in self:
            for subkey in self[key]:
                if isinstance(self[key][subkey], dict):
                    _btype = "individual"
                    break
                elif isinstance(self[key][subkey], (list, np.ndarray)):
                    _btype = "general"
        return _btype

    def start_values(self, order: list = []):
        """Get start values for IVIM parameters."""
        return self._get_boundary(0, order)

    def lower_bounds(self, order: list = []):
        """Get lower stop values for IVIM parameters."""
        return self._get_boundary(1, order)

    def upper_bounds(self, order: list = []):
        """Get upper stop values for IVIM parameters."""
        return self._get_boundary(2, order)

    def _get_boundary(self, pos: int, order: list) -> np.ndarray | dict:
        """Get boundary values for IVIM parameters.

        Shape of boundary values:
            [f1,D1,f2,D2,...,fn,Dn(,TM)] or
            [f1,D1,f2,D2,...,Dn-1,Dn(,TM)] for reduced fitting.
        """
        if self.btype == "general":
            boundaries = dict()
            for key in self:
                boundaries[key] = dict()
                for subkey in self[key]:
                    boundaries[key][subkey] = self[key][subkey][pos]
            _boundaries = []
            for key in order:
                ids = key.split("_")
                _boundaries.append(boundaries[ids[0]][ids[1]])
            _boundaries = np.array(_boundaries)

        elif self.btype == "individual":
            # for pixel by pixel boundaries
            # first get all x0 | lb | ub in a dict
            # second create array of correct order
            boundaries = dict()
            coords = []
            for key in self:
                boundaries[key] = dict()
                for subkey in self[key]:
                    coords = self[key][subkey].keys()
                    for coord in self[key][subkey]:
                        boundaries[key][coord][subkey] = self[key][subkey][coord][pos]
            _boundaries = {}
            for coord in coords:
                for key in order:
                    ids = key.split("_")
                    _boundaries[coord] = boundaries[ids[0]][ids[1]][coord]
        else:
            error_msg = f"Boundary type {self.btype} not recognized."
            logger.error(error_msg)
            raise ValueError(error_msg)
        return _boundaries

    def _set_boundary(self, pos: int, values: list | np.ndarray):
        idx = 0
        for key in self:
            for subkey in self[key]:
                self[key][subkey][pos] = values[idx]
                idx += 1

    def __setitem__(self, key, value):
        """Set value of key in dictionary."""
        super().__setitem__(key, value)
        self._check_boundaries()

    def _check_boundaries(self):
        # check weather the boundaries are correctly set
        for key in self:
            for subkey in self[key]:
                x0 = self[key][subkey][0]
                lb = self[key][subkey][1]
                ub = self[key][subkey][2]
                if not (lb <= x0 <= ub):
                    raise ValueError(
                        f"Start value {x0} is not between bounds {lb} and {ub} for {key}_{subkey}."
                    )

    def get_axis_limits(self) -> tuple:
        """Get Limits for plot axis from parameter values."""
        _min = min(self.lower_bounds)  # this should always be the lowest D value
        d_values = list()
        for key in self["D"]:
            d_values = d_values + self["D"][key]
        _max = max(d_values)

        # _max = max(self.upper_stop_values)
        return _min, _max


class NNLSBoundaryDict(BaseBoundaryDict):
    """Dictionary class for handling NNLS boundaries."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        """Set value of key in dictionary."""
        super().__setitem__(key, value)
        self._check_boundaries()

    def _check_boundaries(self):
        if "d_range" in self:
            d_range = self["d_range"]
            if not (isinstance(d_range, list) and len(d_range) == 2):
                raise ValueError(
                    "d_range must be a list of two elements: [d_min, d_max]."
                )
            if d_range[0] >= d_range[1]:
                raise ValueError("d_range minimum must be less than maximum.")
        if "n_bins" in self:
            n_bins = self["n_bins"]
            if not (isinstance(n_bins, int) and n_bins > 0):
                raise ValueError("n_bins must be a positive integer.")

    def get_axis_limits(self) -> tuple:
        return self.get("d_range", [0])[0], self.get("d_range", [0, 0])[1]
