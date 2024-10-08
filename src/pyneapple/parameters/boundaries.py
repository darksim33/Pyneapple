from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BoundariesBase(ABC):
    """Basic abstract boundaries class"""

    @abstractmethod
    def load(self, _dict: dict):
        """Load dict into class."""
        pass

    @abstractmethod
    def save(self) -> dict:
        """Return dict for saving to json"""
        pass

    @property
    @abstractmethod
    def parameter_names(self) -> list | None:
        """Get list of parameter names."""
        pass

    @property
    @abstractmethod
    def scaling(self):
        """Scaling to parameters id needed."""
        pass

    @abstractmethod
    def apply_scaling(self, value):
        """Apply scaling to parameter values."""
        pass

    @abstractmethod
    def get_axis_limits(self) -> tuple:
        """Get Limits for axis in parameter values."""
        pass


class Boundaries(BoundariesBase):
    def __init__(self):
        self.values = dict()
        self.scaling: str | int | float | list | None = None
        # a factor or string (needs to be added to apply_scaling to boundaries)
        self.dict = dict()
        self.number_points = 250  # reserved for creating spectral array element. behaves like a resolution

    def load(self, _dict: dict):
        self.dict = _dict.copy()

    def save(self):
        _dict = self.dict.copy()
        for key, value in _dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        value[sub_key] = sub_value.tolist()
            if isinstance(value, np.ndarray):
                _dict[key] = value.tolist()
        return _dict

    @property
    def parameter_names(self) -> list | None:
        """Returns parameter names from json for IVIM (and generic names vor NNLS)"""
        return None

    @parameter_names.setter
    def parameter_names(self, data: dict):
        pass

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        self._scaling = value

    def apply_scaling(self, value: list) -> list:
        return value

    def get_axis_limits(self) -> tuple:
        return 0.0001, 1

    def get_boundary_names(self) -> list:
        """Return names of all boundaries as a list."""
        names = []
        for key, value in self.dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    names.append(key + "_" + sub_key)
            else:
                names.append(key)
        return names


class NNLSBoundaries(Boundaries):
    def __init__(self):
        self.scaling = None
        self.dict = dict()
        super().__init__()

    def load(self, data: dict):
        """
        The dictionaries need to be shape according to the following shape:
        "boundaries": {
            "d_range": [],
            "n_bins": []
        }
        Parameters are read starting with the first key descending to bottom level followed by the next key.
        """
        self.dict = data
        self.number_points = data["n_bins"]

    @property
    def parameter_names(self) -> list | None:
        """Returns parameter names for NNLS"""
        names = [f"X{i}" for i in range(0, 10)]
        return names

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        self._scaling = value

    def apply_scaling(self, value: list) -> list:
        """Currently there is no scaling available for NNLS (except CV)."""
        return value

    def get_axis_limits(self) -> tuple:
        return self.dict.get("d_range", [0])[0], self.dict.get("d_range", [1])[1]


class IVIMBoundaries(Boundaries):
    def __init__(self):
        """
        Handle IVIM fitting boundaries.

        Boundaries imported by loading a dict. The dict should have the following structure:
        "D":  {<compartment>: [x0, lb, ub], ... } the compartments should increase from slowest to fastest.
        "f": { <compartment>: [x0, lb, ub], ... } the compartments should increase from slowest to fastest.
        "S": { "0": [x0, lb, ub] }  # S0 is always the last parameter

        Optional:
        "T1": [x0, lb, ub]  # T1 is optional and can be added to the dict.

        Data is imported using the load() method.
        """
        self.dict: dict | None = None
        super().__init__()

    @property
    def parameter_names(self) -> list | None:
        """Returns parameter names from json for IVIM (and generic names vor NNLS)"""
        names = list()
        for key in self.dict:
            for subkey in self.dict[key]:
                names.append(key + "_" + subkey)
        names = self.apply_scaling(names)
        if len(names) == 0:
            names = None
        return names

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        self._scaling = value

    @property
    def start_values(self):
        return self._get_boundary(0)

    @start_values.setter
    def start_values(self, x0: list | np.ndarray):
        self._set_boundary(0, x0)

    @property
    def lower_stop_values(self):
        return self._get_boundary(1)

    @lower_stop_values.setter
    def lower_stop_values(self, lb: list | np.ndarray):
        self._set_boundary(1, lb)

    @property
    def upper_stop_values(self):
        return self._get_boundary(2)

    @upper_stop_values.setter
    def upper_stop_values(self, ub: list | np.ndarray):
        self._set_boundary(2, ub)

    def load(self, data: dict):
        """
        Data Shape:
        "D": {
            "<NAME>": [x0, lb, ub],
            ...
        },
        "f": {
            "<NAME>": [x0, lb, ub],
            ...
        }
        "S": {
            "<NAME>": [x0, lb, ub],
        }
        """
        self.dict = data.copy()

    def save(self) -> dict:
        _dict = super().save()
        return _dict

    def apply_scaling(self, value: list) -> list:
        if isinstance(self._scaling, str):
            if self.scaling == "S/S0" and "S" in self.dict.keys():
                # with S/S0 the number of Parameters is reduced.
                value = value[:-1]
        elif isinstance(self.scaling, (int, float)):
            pass
        return value

    def _get_boundary(self, pos: int):
        # TODO: Remove scale when the fitting dlg is reworked accordingly, adjust dlg accordingly
        values = list()
        for key in self.dict:
            for subkey in self.dict[key]:
                values.append(self.dict[key][subkey][pos])
        values = self.apply_scaling(values)
        values = np.array(values)
        return values

    def _set_boundary(self, pos: int, values: list | np.ndarray):
        idx = 0
        for key in self.dict:
            for subkey in self.dict[key]:
                self.dict[key][subkey][pos] = values[idx]
                idx += 1

    def get_axis_limits(self) -> tuple:
        _min = min(self.lower_stop_values)  # this should always be the lowest D value
        d_values = list()
        for key in self.dict["D"]:
            d_values = d_values + self.dict["D"][key]
        _max = max(d_values)

        # _max = max(self.upper_stop_values)
        return _min, _max
