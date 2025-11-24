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
        If the boundary values are lists or arrays it will be considered as general.
        Which means the same boundaries are used for all pixels.
        If the boundary values are dictionaries it will be considered as individual and
        different boundaries can be used for each pixel.

        Types: "general", "individual"
        """
        _btype = ""
        for key in self:
            for subkey in self[key]:
                if isinstance(self[key][subkey], dict):
                    _btype = "individual"
                    break
                elif isinstance(self[key][subkey], (list, np.ndarray)):
                    _btype = "general"
        return _btype

    def start_values(self, order: list = []) -> np.ndarray | dict:
        """Get start values for IVIM parameters."""
        return self._get_boundary(0, order)

    def lower_bounds(self, order: list = []) -> np.ndarray | dict:
        """Get lower stop values for IVIM parameters."""
        return self._get_boundary(1, order)

    def upper_bounds(self, order: list = []) -> np.ndarray | dict:
        """Get upper stop values for IVIM parameters."""
        return self._get_boundary(2, order)

    def _get_boundary(self, pos: int, order: list) -> np.ndarray | dict:
        """Get boundary values for IVIM parameters.

        Args:
            pos (int): Position of boundary (x0, lb, ub) values to get.
            order (list): Order of parameters to return. (Provided by fit model)

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
                    if not boundaries[key].get(subkey):
                        boundaries[key][subkey] = dict()
                    coords = self[key][subkey].keys()
                    for coord in self[key][subkey]:
                        boundaries[key][subkey][coord] = self[key][subkey][coord][pos]
            _boundaries = {}
            for coord in coords:
                for key in order:
                    ids = key.split("_")
                    if not _boundaries.get(coord):
                        _boundaries[coord] = []
                    _boundaries[coord].append(boundaries[ids[0]][ids[1]][coord])
        else:
            error_msg = f"Boundary type {self.btype} not recognized."
            logger.error(error_msg)
            raise ValueError(error_msg)
        return _boundaries

    def _set_boundary(self, pos: int, values: list | dict | np.ndarray):
        idx = 0
        if self.btype != "general":
            for key in self:
                for subkey in self[key]:
                    self[key][subkey][pos] = values[idx]
                    idx += 1
        elif self.btype == "individual":
            for key in self:
                for subkey in self[key]:
                    for coord in values[key][subkey]:
                        self[key][subkey][coord][pos] = values[key][subkey][coord][pos]
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

    def get_axis_limits(
        self,
    ) -> tuple:
        """Get Limits for plot axis from parameter values."""
        d_values = list()
        for key in self["D"]:
            d_values = d_values + self["D"][key]
        _max = max(d_values)
        _min = min(d_values)
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
