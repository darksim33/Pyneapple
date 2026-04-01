---
name: testing-guidelines
description: 'Conventions for writing pytest tests. Use when: writing new tests, adding fixtures, updating existing tests, running or debugging tests, test naming conventions, parametrize, mocking, markers.'
---

# Testing Guidelines

## When to use me

Use this skill when:
- Writing new tests in `tests/`
- Fixing or updating existing tests
- Adding test fixtures to `conftest.py`
- Running or debugging tests

## Organization

- All tests in `tests/`, named `test_<layer>_<concept>.py` (e.g., `test_solver_nnls.py`)
- Fallback to `test_<module>.py` when no clear layer applies (e.g., `test_io.py`)
- Shared fixtures in `tests/conftest.py`; file/IO fixtures in a dedicated helper module
- Import order: stdlib → third-party (`pytest`, `numpy`) → local (`from mypackage.module import MyClass`)

## Structure

**Class-based tests** (preferred) — group related tests by concept:

```python
class TestMyFeature:
    """Test suite for MyFeature functionality."""

    def test_initialization(self, my_params):
        """MyFeature initializes correctly with valid parameters."""
        obj = MyFeature(*my_params)
        assert obj.n_params == 2
```

**Standalone functions** only for trivial, ungroupable tests.

### Naming

- Methods: `test_<what>_<condition>` — e.g., `test_fit_converges_with_good_initial_values`
- Classes: `Test<Class>` or `Test<Feature>` — e.g., `TestSolverFitting`
- Fixtures: lowercase with type prefix — `solver_params`, `fit_result`, `sample_data`
- **Every test must have a docstring** explaining what is tested

## Fixtures

| Scope | Use for |
|-------|---------|
| `function` (default) | Mutable data, per-test isolation |
| `class` / `module` | Expensive setup shared across a group |
| `session` | File I/O, one-time setup |

Use `yield` for cleanup. Use `tmp_path` for temporary files (auto-cleaned).

### Test Helpers (`tests/test_toolbox.py`)

Initialize helper Tools in `tests/test_toolbox.py`.

## Assertions

- Use plain `assert` (pytest style), never `self.assertEqual()`
- Add messages for non-obvious assertions: `assert x > 0, f"Expected positive, got {x}"`
- NumPy: prefer `np.testing.assert_allclose(a, b, rtol=1e-5)` (detailed diff on failure)
- Exceptions: `with pytest.raises(ValueError, match="must be positive"):`

## Mocking

**Always use `pytest-mock`** (`mocker` fixture), not `unittest.mock`.

```python
def test_with_mock(mocker):
    """Test with mocked dependency."""
    mock_dep = mocker.patch('mypackage.module.dependency')
    mock_dep.return_value = {"status": "success"}
    result = function_under_test()
    mock_dep.assert_called_once()
```

Key patterns: `mocker.patch()`, `mocker.Mock()`, `mocker.spy()`, `mocker.PropertyMock`.

**Mock:** external APIs, expensive computations, time/randomness.
**Don't mock:** the code under test, pure functions, simple data structures.

## Parametrization

Use `@pytest.mark.parametrize` instead of repetitive test functions:

```python
@pytest.mark.parametrize("noise_level", [0.01, 0.05])
def test_fit_noisy_data(self, noise_level):
    """Solver handles noisy data at different levels."""
    ...
```

Use `ids=` for readability when parameter meaning isn't obvious.

## Markers

| Marker | Use for |
|--------|---------|
| `@pytest.mark.slow` | Tests taking more than a few seconds |
| `@pytest.mark.integration` | End-to-end pipeline tests |
| `@pytest.mark.unit` | Isolated tests for individual functions/classes, no external dependencies |
| `@pytest.mark.skip(reason=...)` | Not yet implemented |
| `@pytest.mark.xfail(reason=...)` | Known bugs |

Run selectively: `pytest -m "not slow"`

## Best Practices

1. **Independence** — no shared mutable state between tests; use fixtures
2. **Edge cases** — always test empty inputs, boundary values, error paths
3. **Speed** — small datasets for logic tests; mark slow tests; mock expensive ops
4. **Descriptive names** — `b_values` not `x`, `expected_signal` not `e`
5. **No test logic** — hardcode expected values, don't recalculate in the test
6. **Temporary files** — use `tmp_path` fixture (auto-cleaned by pytest)

## Running Tests

```bash
pytest                           # all tests
pytest tests/test_solver_nnls.py # specific file
pytest -k "model"                # pattern match
pytest -m "not slow"             # exclude markers
pytest -x                        # stop on first failure
pytest --lf                      # rerun last failures
pytest -s                        # print output all print statements
```

## LLM Prompt Template

When using LLMs to generate tests:

```
Create pytest tests for [MODULE_NAME] following these requirements:

1. Class-based organization (Test[ClassName])
2. pytest-mock (mocker fixture) for mocking, NOT unittest.mock
3. Docstrings on all test functions
4. @pytest.mark.parametrize for multiple scenarios
5. Naming: test_<action>_<condition>_<expected>
6. Plain assert statements
7. Independent tests — no shared state
8. Edge cases: empty inputs, boundary values, errors
9. Markers: @pytest.mark.slow, @pytest.mark.gpu as needed
```
