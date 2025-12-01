"""Tests for boundaries module."""

import numpy as np
import pytest

from pyneapple.parameters.boundaries import (
    BaseBoundaryDict,
    IVIMBoundaryDict,
    NNLSBoundaryDict,
)


class TestBaseBoundaryDict:
    """Test suite for BaseBoundaryDict class."""

    def test_initialization_empty(self):
        """Test initialization of empty BaseBoundaryDict."""
        boundary = BaseBoundaryDict()
        assert isinstance(boundary, dict)
        assert len(boundary) == 0

    def test_initialization_with_args(self):
        """Test initialization with arguments."""
        boundary = BaseBoundaryDict({"key1": "value1", "key2": "value2"})
        assert len(boundary) == 2
        assert boundary["key1"] == "value1"
        assert boundary["key2"] == "value2"

    def test_initialization_with_kwargs(self):
        """Test initialization with keyword arguments."""
        boundary = BaseBoundaryDict(key1="value1", key2="value2")
        assert len(boundary) == 2
        assert boundary["key1"] == "value1"
        assert boundary["key2"] == "value2"

    def test_get_axis_limits_default(self):
        """Test default axis limits."""
        boundary = BaseBoundaryDict()
        limits = boundary.get_axis_limits()
        assert limits == (0.0001, 1)
        assert isinstance(limits, tuple)
        assert len(limits) == 2

    def test_parameter_names_empty(self):
        """Test parameter_names property with empty dict."""
        boundary = BaseBoundaryDict()
        names = boundary.parameter_names
        assert isinstance(names, list)
        assert len(names) == 0

    def test_parameter_names_simple(self):
        """Test parameter_names property with simple values."""
        boundary = BaseBoundaryDict({"param1": 1.0, "param2": 2.0})
        names = boundary.parameter_names
        assert isinstance(names, list)
        assert len(names) == 2
        assert "param1" in names
        assert "param2" in names

    def test_parameter_names_nested(self):
        """Test parameter_names property with nested dict values."""
        boundary = BaseBoundaryDict(
            {
                "f": {"component1": 0.5, "component2": 0.3},
                "D": {"component1": 0.001, "component2": 0.002},
            }
        )
        names = boundary.parameter_names
        assert isinstance(names, list)
        assert len(names) == 4
        assert "f_component1" in names
        assert "f_component2" in names
        assert "D_component1" in names
        assert "D_component2" in names

    def test_parameter_names_mixed(self):
        """Test parameter_names property with mixed simple and nested values."""
        boundary = BaseBoundaryDict(
            {"simple_param": 1.0, "nested": {"sub1": 0.5, "sub2": 0.3}}
        )
        names = boundary.parameter_names
        assert isinstance(names, list)
        assert len(names) == 3
        assert "simple_param" in names
        assert "nested_sub1" in names
        assert "nested_sub2" in names

    def test_parameter_names_order(self):
        """Test that parameter_names preserves order."""
        boundary = BaseBoundaryDict()
        boundary["first"] = {"a": 1, "b": 2}
        boundary["second"] = 3
        boundary["third"] = {"c": 4}

        names = boundary.parameter_names
        # Check that nested params come after their parent in order
        first_a_idx = names.index("first_a")
        first_b_idx = names.index("first_b")
        second_idx = names.index("second")
        third_c_idx = names.index("third_c")

        assert first_a_idx < second_idx
        assert first_b_idx < second_idx
        assert second_idx < third_c_idx

    def test_dict_operations(self):
        """Test that standard dict operations work."""
        boundary = BaseBoundaryDict()
        boundary["key1"] = "value1"

        assert "key1" in boundary
        assert boundary["key1"] == "value1"
        assert len(boundary) == 1

        del boundary["key1"]
        assert "key1" not in boundary
        assert len(boundary) == 0

    def test_update_method(self):
        """Test dict update method."""
        boundary = BaseBoundaryDict({"a": 1})
        boundary.update({"b": 2, "c": 3})

        assert len(boundary) == 3
        assert boundary["a"] == 1
        assert boundary["b"] == 2
        assert boundary["c"] == 3


class TestIVIMBoundaryDict:
    """Test suite for IVIMBoundaryDict class."""

    def test_initialization_empty(self):
        """Test initialization of empty IVIMBoundaryDict."""
        boundary = IVIMBoundaryDict()
        assert isinstance(boundary, dict)
        assert isinstance(boundary, BaseBoundaryDict)
        assert len(boundary) == 0

    def test_btype_general(self):
        """Test btype property for general boundaries."""
        boundary = IVIMBoundaryDict(
            {
                "f": {"comp1": [0.2, 0.0, 1.0], "comp2": [0.1, 0.0, 1.0]},
                "D": {"comp1": [0.001, 0.0001, 0.003], "comp2": [0.002, 0.0001, 0.005]},
            }
        )
        assert boundary.btype == "general"

    def test_btype_individual(self):
        """Test btype property for individual (pixel-by-pixel) boundaries."""
        boundary = IVIMBoundaryDict(
            {
                "f": {"comp1": {(0, 0): [0.2, 0.0, 1.0], (0, 1): [0.3, 0.0, 1.0]}},
                "D": {
                    "comp1": {
                        (0, 0): [0.001, 0.0001, 0.003],
                        (0, 1): [0.002, 0.0001, 0.005],
                    }
                },
            }
        )
        assert boundary.btype == "individual"

    def test_start_values_general(self):
        """Test start_values method for general boundaries."""
        boundary = IVIMBoundaryDict(
            {
                "f": {"comp1": [0.2, 0.0, 1.0], "comp2": [0.1, 0.0, 1.0]},
                "D": {"comp1": [0.001, 0.0001, 0.003], "comp2": [0.002, 0.0001, 0.005]},
            }
        )
        order = ["f_comp1", "D_comp1", "f_comp2", "D_comp2"]
        start = boundary.start_values(order)

        assert isinstance(start, np.ndarray)
        np.testing.assert_array_equal(start, [0.2, 0.001, 0.1, 0.002])

    def test_lower_bounds_general(self):
        """Test lower_bounds method for general boundaries."""
        boundary = IVIMBoundaryDict(
            {
                "f": {"comp1": [0.2, 0.0, 1.0], "comp2": [0.1, 0.0, 1.0]},
                "D": {"comp1": [0.001, 0.0001, 0.003], "comp2": [0.002, 0.0001, 0.005]},
            }
        )
        order = ["f_comp1", "D_comp1", "f_comp2", "D_comp2"]
        lower = boundary.lower_bounds(order)

        assert isinstance(lower, np.ndarray)
        np.testing.assert_array_equal(lower, [0.0, 0.0001, 0.0, 0.0001])

    def test_upper_bounds_general(self):
        """Test upper_bounds method for general boundaries."""
        boundary = IVIMBoundaryDict(
            {
                "f": {"comp1": [0.2, 0.0, 1.0], "comp2": [0.1, 0.0, 1.0]},
                "D": {"comp1": [0.001, 0.0001, 0.003], "comp2": [0.002, 0.0001, 0.005]},
            }
        )
        order = ["f_comp1", "D_comp1", "f_comp2", "D_comp2"]
        upper = boundary.upper_bounds(order)

        assert isinstance(upper, np.ndarray)
        np.testing.assert_array_equal(upper, [1.0, 0.003, 1.0, 0.005])

    def test_boundary_order_flexibility(self):
        """Test that boundary methods respect different parameter orders."""
        boundary = IVIMBoundaryDict(
            {
                "f": {"comp1": [0.2, 0.0, 1.0], "comp2": [0.1, 0.0, 1.0]},
                "D": {"comp1": [0.001, 0.0001, 0.003], "comp2": [0.002, 0.0001, 0.005]},
            }
        )

        order1 = ["f_comp1", "D_comp1", "f_comp2", "D_comp2"]
        order2 = ["D_comp1", "f_comp1", "D_comp2", "f_comp2"]

        start1 = boundary.start_values(order1)
        start2 = boundary.start_values(order2)

        np.testing.assert_array_equal(start1, [0.2, 0.001, 0.1, 0.002])
        np.testing.assert_array_equal(start2, [0.001, 0.2, 0.002, 0.1])

    def test_check_boundaries_valid(self):
        """Test that valid boundaries pass validation."""
        boundary = IVIMBoundaryDict()
        # This should not raise an error
        boundary["f"] = {"comp1": [0.2, 0.0, 1.0]}
        boundary["D"] = {"comp1": [0.001, 0.0001, 0.003]}

    def test_check_boundaries_invalid_start_too_low(self):
        """Test that start value below lower bound raises ValueError."""
        boundary = IVIMBoundaryDict()
        with pytest.raises(ValueError, match="Start value .* is not between bounds"):
            boundary["f"] = {"comp1": [-0.1, 0.0, 1.0]}

    def test_check_boundaries_invalid_start_too_high(self):
        """Test that start value above upper bound raises ValueError."""
        boundary = IVIMBoundaryDict()
        with pytest.raises(ValueError, match="Start value .* is not between bounds"):
            boundary["f"] = {"comp1": [1.5, 0.0, 1.0]}

    def test_check_boundaries_edge_cases(self):
        """Test that start values at boundary edges are valid."""
        boundary = IVIMBoundaryDict()
        # Start at lower bound
        boundary["f"] = {"comp1": [0.0, 0.0, 1.0]}
        assert boundary["f"]["comp1"] == [0.0, 0.0, 1.0]

        # Start at upper bound
        boundary["f"]["comp2"] = [1.0, 0.0, 1.0]
        assert boundary["f"]["comp2"] == [1.0, 0.0, 1.0]

    def test_get_axis_limits(self):
        """Test get_axis_limits method."""
        boundary = IVIMBoundaryDict(
            {
                "f": {"comp1": [0.2, 0.0, 1.0]},
                "D": {"comp1": [0.001, 0.0001, 0.003], "comp2": [0.005, 0.0002, 0.01]},
            }
        )
        limits = boundary.get_axis_limits()
        assert isinstance(limits, tuple)
        assert len(limits) == 2
        # Min should be lowest lower bound (0.0001)
        assert limits[0] == 0.0001
        # Max should be highest D value (0.01)
        assert limits[1] == 0.01

    def test_get_axis_limits_multiple_d_components(self):
        """Test axis limits with multiple D components."""
        boundary = IVIMBoundaryDict(
            {
                "D": {
                    "comp1": [0.001, 0.0001, 0.003],
                    "comp2": [0.005, 0.0002, 0.01],
                    "comp3": [0.015, 0.0001, 0.02],
                }
            }
        )

        limits = boundary.get_axis_limits()
        assert limits[0] == 0.0001
        assert limits[1] == 0.02

    def test_get_boundary_invalid_type(self):
        """Test that invalid boundary type raises ValueError."""
        boundary = IVIMBoundaryDict()
        # Manually set an invalid structure to trigger the error

        with pytest.raises(ValueError):
            boundary["f"] = {"comp1": "invalid_type"}

    def test_empty_order_list(self):
        """Test boundary methods with empty order list."""
        boundary = IVIMBoundaryDict(
            {"f": {"comp1": [0.2, 0.0, 1.0]}, "D": {"comp1": [0.001, 0.0001, 0.003]}}
        )

        start = boundary.start_values([])
        assert isinstance(start, np.ndarray)
        assert len(start) == 0

    def test_numpy_array_boundaries(self):
        """Test that numpy arrays work as boundary values."""
        boundary = IVIMBoundaryDict(
            {
                "f": {"comp1": np.array([0.2, 0.0, 1.0])},
                "D": {"comp1": np.array([0.001, 0.0001, 0.003])},
            }
        )

        assert boundary.btype == "general"
        order = ["f_comp1", "D_comp1"]
        start = boundary.start_values(order)
        np.testing.assert_array_equal(start, [0.2, 0.001])

    def test_inheritance_from_base(self):
        """Test that IVIMBoundaryDict inherits from BaseBoundaryDict."""
        boundary = IVIMBoundaryDict(
            {"f": {"comp1": [0.2, 0.0, 1.0]}, "D": {"comp1": [0.001, 0.0001, 0.003]}}
        )

        # Should have parameter_names property from base class
        names = boundary.parameter_names
        assert "f_comp1" in names
        assert "D_comp1" in names


class TestIVIMBoundaryDictWithModelConfigurations:
    """Test suite for IVIMBoundaryDict with different model configurations."""

    def test_boundary_length_monoexp_default(self):
        """Test boundary length for MonoExp with default settings (fit_reduced=True)."""
        from pyneapple.models import MonoExpFitModel

        model = MonoExpFitModel()
        # Default: fit_reduced=True, fit_t1=False -> args: [D_1, S_0]
        expected_args = model.args

        boundary = IVIMBoundaryDict(
            {
                "D": {"1": [0.001, 0.0001, 0.003]},
                "S": {"0": [100, 10, 500]},
            }
        )

        start = boundary.start_values(expected_args)
        lower = boundary.lower_bounds(expected_args)
        upper = boundary.upper_bounds(expected_args)

        assert len(start) == len(expected_args)
        assert len(lower) == len(expected_args)
        assert len(upper) == len(expected_args)

    def test_boundary_length_monoexp_with_t1(self):
        """Test boundary length for MonoExp with T1 fitting enabled."""
        from pyneapple.models import MonoExpFitModel

        model = MonoExpFitModel()
        model.fit_t1 = True
        model.repetition_time = 50.0
        # fit_reduced=True, fit_t1=True -> args: [D_1, S_0, T_1]
        expected_args = model.args

        boundary = IVIMBoundaryDict(
            {
                "D": {"1": [0.001, 0.0001, 0.003]},
                "S": {"0": [100, 10, 500]},
                "T": {"1": [1000, 500, 3000]},
            }
        )

        start = boundary.start_values(expected_args)
        lower = boundary.lower_bounds(expected_args)
        upper = boundary.upper_bounds(expected_args)

        assert len(start) == len(expected_args)
        assert len(lower) == len(expected_args)
        assert len(upper) == len(expected_args)
        assert len(start) == 3  # D_1, S_0, T_1

    def test_boundary_length_monoexp_fit_reduced(self):
        """Test boundary length for MonoExp with fit_reduced=False."""
        from pyneapple.models import MonoExpFitModel

        model = MonoExpFitModel()
        model.fit_reduced = True
        # fit_reduced=True, fit_t1=False -> args: [D_1]
        expected_args = model.args

        boundary = IVIMBoundaryDict(
            {
                "D": {"1": [0.001, 0.0001, 0.003]},
                "S": {"0": [100, 10, 500]},
            }
        )

        start = boundary.start_values(expected_args)
        lower = boundary.lower_bounds(expected_args)
        upper = boundary.upper_bounds(expected_args)

        assert len(start) == len(expected_args)
        assert len(lower) == len(expected_args)
        assert len(upper) == len(expected_args)
        assert len(start) == 1  # D_1

    def test_boundary_length_monoexp_fit_reduced_false(self):
        """Test boundary length for MonoExp with fit_reduced=False."""
        from pyneapple.models import MonoExpFitModel

        model = MonoExpFitModel()
        model.fit_reduced = False
        # fit_reduced=False, fit_t1=False -> args: [D_1, S_0]
        expected_args = model.args

        boundary = IVIMBoundaryDict(
            {
                "D": {"1": [0.001, 0.0001, 0.003]},
                "S": {"0": [100, 10, 500]},
            }
        )

        start = boundary.start_values(expected_args)
        lower = boundary.lower_bounds(expected_args)
        upper = boundary.upper_bounds(expected_args)

        assert len(start) == len(expected_args)
        assert len(lower) == len(expected_args)
        assert len(upper) == len(expected_args)
        assert len(start) == 2  # D_1, S_0

    def test_boundary_length_biexp_default(self):
        """Test boundary length for BiExp with default settings."""
        from pyneapple.models import BiExpFitModel

        model = BiExpFitModel()
        # Default: fit_reduced=True, fit_S0=False, fit_t1=False -> args: [f_1, D_1, D_2]
        expected_args = model.args

        boundary = IVIMBoundaryDict(
            {
                "f": {"1": [0.2, 0.0, 1.0], "2": [0.2, 0.0, 1.0]},
                "D": {"1": [0.001, 0.0001, 0.003], "2": [0.01, 0.005, 0.05]},
            }
        )

        start = boundary.start_values(expected_args)
        lower = boundary.lower_bounds(expected_args)
        upper = boundary.upper_bounds(expected_args)

        assert len(start) == len(expected_args)
        assert len(lower) == len(expected_args)
        assert len(upper) == len(expected_args)
        assert len(start) == 4  # f_1, D_1, f_2, D_2

    def test_boundary_length_biexp_with_s0(self):
        """Test boundary length for BiExp with fit_S0=True."""
        from pyneapple.models import BiExpFitModel

        model = BiExpFitModel(fit_S0=True)
        model.fit_reduced = False
        # fit_reduced=False, fit_S0=True -> args: [f_1, D_1, f_2, D_2, S_0]
        expected_args = model.args

        boundary = IVIMBoundaryDict(
            {
                "f": {"1": [0.2, 0.0, 1.0]},
                "D": {"1": [0.001, 0.0001, 0.003], "2": [0.01, 0.005, 0.05]},
                "S": {"0": [100, 10, 500]},
            }
        )

        start = boundary.start_values(expected_args)
        lower = boundary.lower_bounds(expected_args)
        upper = boundary.upper_bounds(expected_args)

        assert len(start) == len(expected_args)
        assert len(lower) == len(expected_args)
        assert len(upper) == len(expected_args)
        assert len(start) == 4  # f_1, D_1, D_2, S_0

    def test_boundary_length_biexp_fit_reduced_True(self):
        """Test boundary length for BiExp with fit_reduced=False."""
        from pyneapple.models import BiExpFitModel

        model = BiExpFitModel()
        model.fit_reduced = True
        # fit_reduced=False, fit_S0=False -> args: [f_1, D_1, f_2, D_2]
        expected_args = model.args

        boundary = IVIMBoundaryDict(
            {
                "f": {"1": [0.2, 0.0, 1.0], "2": [0.3, 0.0, 1.0]},
                "D": {"1": [0.001, 0.0001, 0.003], "2": [0.01, 0.005, 0.05]},
            }
        )

        start = boundary.start_values(expected_args)
        lower = boundary.lower_bounds(expected_args)
        upper = boundary.upper_bounds(expected_args)

        assert len(start) == len(expected_args)
        assert len(lower) == len(expected_args)
        assert len(upper) == len(expected_args)
        assert len(start) == 3  # f_1, D_1, D_2

    def test_boundary_length_biexp_fit_reduced_false(self):
        """Test boundary length for BiExp with fit_reduced=False."""
        from pyneapple.models import BiExpFitModel

        model = BiExpFitModel()
        model.fit_reduced = False
        # fit_reduced=False, fit_S0=False -> args: [f_1, D_1, f_2, D_2]
        expected_args = model.args

        boundary = IVIMBoundaryDict(
            {
                "f": {"1": [0.2, 0.0, 1.0], "2": [0.3, 0.0, 1.0]},
                "D": {"1": [0.001, 0.0001, 0.003], "2": [0.01, 0.005, 0.05]},
            }
        )

        start = boundary.start_values(expected_args)
        lower = boundary.lower_bounds(expected_args)
        upper = boundary.upper_bounds(expected_args)

        assert len(start) == len(expected_args)
        assert len(lower) == len(expected_args)
        assert len(upper) == len(expected_args)
        assert len(start) == 4  # f_1, D_1, f_2, D_2

    def test_boundary_length_biexp_with_t1(self):
        """Test boundary length for BiExp with T1 fitting enabled."""
        from pyneapple.models import BiExpFitModel

        model = BiExpFitModel()
        model.fit_reduced = True
        model.fit_t1 = True
        model.repetition_time = 50.0
        # fit_reduced=True, fit_t1=True -> args: [f_1, D_1, D_2, T_1]
        expected_args = model.args

        boundary = IVIMBoundaryDict(
            {
                "f": {"1": [0.2, 0.0, 1.0]},
                "D": {"1": [0.001, 0.0001, 0.003], "2": [0.01, 0.005, 0.05]},
                "T": {"1": [1000, 500, 3000]},
            }
        )

        start = boundary.start_values(expected_args)
        lower = boundary.lower_bounds(expected_args)
        upper = boundary.upper_bounds(expected_args)

        assert len(start) == len(expected_args)
        assert len(lower) == len(expected_args)
        assert len(upper) == len(expected_args)
        assert len(start) == 4  # f_1, D_1, D_2, T_1

    def test_boundary_length_triexp_default(self):
        """Test boundary length for TriExp with default settings."""
        from pyneapple.models import TriExpFitModel

        model = TriExpFitModel()
        # Default: fit_reduced=True, fit_t1=False -> args: [f_1, D_1, f_2, D_2, D_3]
        expected_args = model.args

        boundary = IVIMBoundaryDict(
            {
                "f": {"1": [0.2, 0.0, 1.0], "2": [0.3, 0.0, 1.0], "3": [0.5, 0.0, 1.0]},
                "D": {
                    "1": [0.001, 0.0001, 0.003],
                    "2": [0.01, 0.005, 0.05],
                    "3": [0.1, 0.05, 0.2],
                },
            }
        )

        start = boundary.start_values(expected_args)
        lower = boundary.lower_bounds(expected_args)
        upper = boundary.upper_bounds(expected_args)

        assert len(start) == len(expected_args)
        assert len(lower) == len(expected_args)
        assert len(upper) == len(expected_args)
        assert len(start) == 6  # f_1, D_1, f_2, D_2, f_3, D_3

    def test_boundary_length_triexp_with_t1(self):
        """Test boundary length for TriExp with T1 fitting enabled."""
        from pyneapple.models import TriExpFitModel

        model = TriExpFitModel()
        model.fit_reduced = True
        model.fit_t1 = True
        model.repetition_time = 50.0
        # fit_reduced=True, fit_t1=True -> args: [f_1, D_1, f_2, D_2, D_3, T_1]
        expected_args = model.args

        boundary = IVIMBoundaryDict(
            {
                "f": {"1": [0.2, 0.0, 1.0], "2": [0.3, 0.0, 1.0]},
                "D": {
                    "1": [0.001, 0.0001, 0.003],
                    "2": [0.01, 0.005, 0.05],
                    "3": [0.1, 0.05, 0.2],
                },
                "T": {"1": [1000, 500, 3000]},
            }
        )

        start = boundary.start_values(expected_args)
        lower = boundary.lower_bounds(expected_args)
        upper = boundary.upper_bounds(expected_args)

        assert len(start) == len(expected_args)
        assert len(lower) == len(expected_args)
        assert len(upper) == len(expected_args)
        assert len(start) == 6  # f_1, D_1, f_2, D_2, D_3, T_1

    def test_boundary_length_individual_boundaries_with_configs(self):
        """Test boundary length with individual boundaries and different model configs."""
        from pyneapple.models import BiExpFitModel

        model = BiExpFitModel(fit_S0=True)
        model.fit_reduced = False
        model.fit_t1 = True
        model.fit_S0
        model.mixing_time = 50.0
        # fit_reduced=False, fit_S0=True, fit_t1=True -> args: [f_1, D_1, f_2, D_2, S_0, T_1]
        expected_args = model.args

        boundary = IVIMBoundaryDict(
            {
                "f": {
                    "1": {(0, 0): [0.2, 0.0, 1.0]},
                },
                "D": {
                    "1": {(0, 0): [0.001, 0.0001, 0.003]},
                    "2": {(0, 0): [0.01, 0.005, 0.05]},
                },
                "S": {"0": {(0, 0): [100, 10, 500]}},
                "T": {"1": {(0, 0): [1000, 500, 3000]}},
            }
        )

        start = boundary.start_values(expected_args)
        lower = boundary.lower_bounds(expected_args)
        upper = boundary.upper_bounds(expected_args)

        # For individual boundaries, these are dicts
        assert isinstance(start, dict)
        assert isinstance(lower, dict)
        assert isinstance(upper, dict)

        # Check length for first coordinate
        coord = (0, 0)
        assert len(start[coord]) == len(expected_args)
        assert len(lower[coord]) == len(expected_args)
        assert len(upper[coord]) == len(expected_args)
        assert len(start[coord]) == 5  # f_1, D_1, D_2, S_0, T_1
