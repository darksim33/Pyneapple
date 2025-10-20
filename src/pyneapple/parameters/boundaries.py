"""Module for handling boundaries for fitting parameters."""

from __future__ import annotations

import numpy as np


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
    def start_values(self):
        """Get start values for IVIM parameters."""
        return self._get_boundary(0)

    @start_values.setter
    def start_values(self, x0: list | np.ndarray):
        self._set_boundary(0, x0)

    @property
    def lower_bounds(self):
        """Get lower stop values for IVIM parameters."""
        return self._get_boundary(1)

    @lower_bounds.setter
    def lower_bounds(self, lb: list | np.ndarray):
        self._set_boundary(1, lb)

    @property
    def upper_bounds(self):
        """Get upper stop values for IVIM parameters."""
        return self._get_boundary(2)

    @upper_bounds.setter
    def upper_bounds(self, ub: list | np.ndarray):
        self._set_boundary(2, ub)

    def _get_boundary(self, pos: int) -> np.ndarray:
        """Get boundary values for IVIM parameters.

        Shape of boundary values:
            [f1,D1,f2,D2,...,fn,Dn(,TM)] or
            [f1,D1,f2,D2,...,Dn-1,Dn(,TM)] for reduced fitting.
        """
        d_subkeys = list(self.get("D", {}).keys())
        f_subkeys = list(self.get("f", {}).keys())
        # Preserve order while getting unique values
        unique_subkeys = []
        seen = set()
        for key in f_subkeys + d_subkeys:
            if key not in seen:
                unique_subkeys.append(key)
                seen.add(key)
        # Get keys that are not 'D' or 'f'
        other_keys = [key for key in self.keys() if key not in ["D", "f"]]
        boundaries = list()

        # Add f and D values
        for subkey in unique_subkeys:
            if subkey in f_subkeys:
                boundaries.append(self["f"][subkey][pos])
            if subkey in d_subkeys:
                boundaries.append(self["D"][subkey][pos])

        for key in other_keys:
            for subkey in self[key].keys():
                boundaries.append(self[key][subkey][pos])
        return np.array(boundaries)

    def _set_boundary(self, pos: int, values: list | np.ndarray):
        idx = 0
        for key in self:
            for subkey in self[key]:
                self[key][subkey][pos] = values[idx]
                idx += 1

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

    def get_axis_limits(self) -> tuple:
        return self.get("d_range", [0])[0], self.get("d_range", [0, 0])[1]
