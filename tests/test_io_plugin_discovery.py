"""Tests for entry-point based plugin discovery in io/toml.py."""

from __future__ import annotations

from importlib.metadata import EntryPoint

import pytest

from pyneapple.io.toml import _discover_plugins, _resolve


class TestDiscoverPlugins:
    """Test suite for _discover_plugins() registry population."""

    def test_adds_new_entry_point(self, mocker):
        """New entry-point names are added to the registry."""
        mock_ep = mocker.Mock(spec=EntryPoint)
        mock_ep.name = "gpufit_curvefit"
        mock_ep.value = "pyneapple_gpufit.solvers:GpuCurveFitSolver"
        mocker.patch(
            "pyneapple.io.toml.entry_points",
            return_value=[mock_ep],
        )

        registry: dict = {}
        _discover_plugins("pyneapple.solvers", registry)

        assert "gpufit_curvefit" in registry
        assert registry["gpufit_curvefit"] is mock_ep

    def test_does_not_overwrite_builtin(self, mocker):
        """Built-in registry entries are never replaced by plugins."""
        sentinel = object()
        mock_ep = mocker.Mock(spec=EntryPoint)
        mock_ep.name = "curvefit"
        mock_ep.value = "some.package:SomeClass"
        mocker.patch(
            "pyneapple.io.toml.entry_points",
            return_value=[mock_ep],
        )

        registry: dict = {"curvefit": sentinel}
        _discover_plugins("pyneapple.solvers", registry)

        assert registry["curvefit"] is sentinel

    def test_empty_when_no_plugins_installed(self, mocker):
        """Registry remains unchanged when no entry-points are registered."""
        mocker.patch("pyneapple.io.toml.entry_points", return_value=[])
        registry: dict = {}
        _discover_plugins("pyneapple.solvers", registry)
        assert registry == {}

    def test_multiple_plugins_added(self, mocker):
        """All discovered entry-points are added in one call."""
        eps = []
        for name in ("plugin_a", "plugin_b", "plugin_c"):
            ep = mocker.Mock(spec=EntryPoint)
            ep.name = name
            ep.value = f"some.package:{name}"
            eps.append(ep)

        mocker.patch("pyneapple.io.toml.entry_points", return_value=eps)

        registry: dict = {}
        _discover_plugins("pyneapple.solvers", registry)

        assert set(registry) == {"plugin_a", "plugin_b", "plugin_c"}


class TestResolve:
    """Test suite for _resolve() lazy-loading helper."""

    def test_returns_plain_class_unchanged(self):
        """A plain class in the registry is returned as-is without calling load()."""

        class DummySolver:
            pass

        registry = {"dummy": DummySolver}
        result = _resolve(registry, "dummy")
        assert result is DummySolver

    def test_loads_entry_point_on_first_access(self, mocker):
        """An EntryPoint is loaded lazily when first resolved."""

        class FakeSolver:
            pass

        mock_ep = mocker.Mock(spec=EntryPoint)
        mock_ep.load.return_value = FakeSolver
        registry = {"gpusolver": mock_ep}

        result = _resolve(registry, "gpusolver")

        mock_ep.load.assert_called_once()
        assert result is FakeSolver

    def test_caches_loaded_class(self, mocker):
        """After the first load() call the class is cached; load() is not called again."""

        class FakeSolver:
            pass

        mock_ep = mocker.Mock(spec=EntryPoint)
        mock_ep.load.return_value = FakeSolver
        registry = {"gpusolver": mock_ep}

        _resolve(registry, "gpusolver")
        _resolve(registry, "gpusolver")

        mock_ep.load.assert_called_once()

    def test_raises_key_error_for_unknown_key(self):
        """KeyError is raised for a key not present in the registry."""
        with pytest.raises(KeyError):
            _resolve({}, "nonexistent")


class TestBuildFitterWithPlugin:
    """Integration: build_fitter() resolves a plugin solver via EntryPoint."""

    def test_plugin_solver_used_in_build_fitter(self, mocker, tmp_path):
        """build_fitter() can instantiate a fitter that uses a plugin solver."""
        from pyneapple.io.toml import _SOLVER_REGISTRY, FittingConfig
        from pyneapple.solvers import CurveFitSolver

        # Patch the solver registry so "plugin_solver" is available as an EP
        mock_ep = mocker.Mock(spec=EntryPoint)
        mock_ep.load.return_value = CurveFitSolver  # reuse a real solver for simplicity
        _SOLVER_REGISTRY["plugin_solver"] = mock_ep

        try:
            config = FittingConfig(
                fitter_type="pixelwise",
                model_type="monoexp",
                solver_type="plugin_solver",
                solver_kwargs={"max_iter": 250, "tol": 1e-8},
                p0={"S0": 1.0, "D": 0.001},
                bounds={"S0": (0.0, 2.0), "D": (0.0, 0.1)},
            )
            fitter = config.build_fitter()

            mock_ep.load.assert_called_once()
            assert isinstance(fitter.solver, CurveFitSolver)
        finally:
            # Restore registry state regardless of test outcome
            _SOLVER_REGISTRY.pop("plugin_solver", None)
