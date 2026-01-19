# Testing Guidelines

This document provides comprehensive guidelines for writing tests in the Pyneapple project. Follow these conventions when creating new tests or working with LLMs to generate test code.

## Table of Contents

1. [Test Organization](#test-organization)
2. [Test Structure](#test-structure)
3. [Fixtures](#fixtures)
4. [Mocking](#mocking)
5. [Assertions](#assertions)
6. [Naming Conventions](#naming-conventions)
7. [Docstrings](#docstrings)
8. [Parametrization](#parametrization)
9. [Test Markers](#test-markers)
10. [Best Practices](#best-practices)

---

## Test Organization

### File Structure

- All test files must be located in the `tests/` directory
- Test files should be named `test_<module_name>.py`
- Each test file should correspond to a source module in `src/pyneapple/`
- Shared fixtures belong in `tests/conftest.py` or `tests/_files.py`

### Module Organization

```python
# Standard library imports
import json
from pathlib import Path

# Third-party imports
import pytest
import numpy as np

# Local imports
from pyneapple.models import IVIMModel
from pyneapple.parameters import IVIMParameters
```

---

## Test Structure

### Class-Based Tests (Preferred)

Use class-based tests for organizing related test cases. Classes provide better grouping and can share setup logic.

```python
class TestIVIMModel:
    """Test suite for IVIM model functionality."""

    def test_model_initialization(self, ivim_params):
        """Test that IVIM model initializes correctly with valid parameters."""
        model = IVIMModel(ivim_params)
        assert model is not None
        assert model.n_params == 3

    def test_model_evaluation(self, ivim_params, b_values):
        """Test that model correctly evaluates signal for given parameters."""
        model = IVIMModel(ivim_params)
        signal = model.evaluate(b_values, [0.8, 0.2, 0.003])
        assert signal.shape == b_values.shape
        assert np.all(signal > 0)
```

### Function-Based Tests (Simple Cases Only)

Use standalone functions only for very simple, isolated unit tests that don't benefit from grouping:

```python
def test_version_string():
    """Test that version string is correctly formatted."""
    from pyneapple import __version__
    assert isinstance(__version__, str)
    assert len(__version__.split('.')) >= 2
```

### Test Method Naming

- Test methods must start with `test_`
- Use descriptive names that explain what is being tested
- Follow pattern: `test_<action>_<expected_outcome>`

```python
def test_fit_converges_with_good_initial_values(self):
    """Good example: clear what is tested and expected."""
    pass

def test_fit(self):
    """Bad example: too vague."""
    pass
```

---

## Fixtures

### Fixture Scope

Specify fixture scope explicitly based on usage:

- `function` (default): New instance for each test function
- `class`: Shared across all methods in a test class
- `module`: Shared across all tests in a file
- `session`: Shared across entire test session

```python
@pytest.fixture(scope="function")
def temp_data():
    """Temporary data recreated for each test."""
    return np.random.rand(10, 10)

@pytest.fixture(scope="session")
def sample_nifti_file(tmp_path_factory):
    """Expensive file I/O shared across all tests."""
    filepath = tmp_path_factory.mktemp("data") / "sample.nii"
    # Create file once
    return filepath
```

### Fixture Location

- **Shared fixtures**: Place in `conftest.py` (auto-discovery)
- **File/I/O fixtures**: Place in `tests/_files.py` 
- **Module-specific fixtures**: Define in the test file itself

### Fixture Naming

- Use descriptive, lowercase names with underscores
- Prefix with module name for clarity: `ivim_params`, `nnls_result`
- Avoid generic names like `data`, `result`, `obj`

```python
# Good
@pytest.fixture
def ivim_biexp_parameters():
    """IVIM parameters for biexponential model."""
    return IVIMParameters.from_file("params_biexp.json")

# Bad
@pytest.fixture
def params():
    """Some parameters."""
    return IVIMParameters.from_file("params_biexp.json")
```

### Fixture Cleanup

Use `yield` for fixtures requiring cleanup:

```python
@pytest.fixture
def output_file(tmp_path):
    """Create temporary output file and clean up after test."""
    filepath = tmp_path / "output.h5"
    yield filepath
    # Cleanup after test completes
    if filepath.exists():
        filepath.unlink()
```

---

## Mocking

### Use pytest-mock (mocker fixture)

**Always use `pytest-mock` (mocker fixture) instead of `unittest.mock`**. It provides better integration with pytest and cleaner syntax.

#### Installation

```bash
pip install pytest-mock
```

### Basic Mocking Patterns

#### Patching Functions

```python
def test_function_with_external_dependency(mocker):
    """Test function that calls an external API."""
    # Mock the external call
    mock_api = mocker.patch('pyneapple.utils.api_client.fetch_data')
    mock_api.return_value = {"status": "success"}
    
    result = process_api_data()
    assert result["status"] == "success"
    mock_api.assert_called_once()
```

#### Patching Methods

```python
def test_method_calls_logger(mocker):
    """Test that method logs information correctly."""
    mock_logger = mocker.patch('pyneapple.utils.logger.log')
    
    obj = MyClass()
    obj.process()
    
    mock_logger.assert_called_with("Processing complete", level="INFO")
```

#### Creating Mock Objects

```python
def test_with_mock_object(mocker):
    """Test using a mock object with specific attributes."""
    mock_params = mocker.Mock()
    mock_params.n_params = 3
    mock_params.bounds = [(0, 1), (0, 1), (0, 0.01)]
    
    model = IVIMModel(mock_params)
    assert model.n_params == 3
```

#### Property Mocking

```python
def test_property_access(mocker):
    """Test accessing a mocked property."""
    mock_obj = mocker.Mock()
    mocker.patch.object(
        type(mock_obj), 
        'expensive_property', 
        new_callable=mocker.PropertyMock,
        return_value=42
    )
    
    assert mock_obj.expensive_property == 42
```

#### Spy on Real Methods

```python
def test_spy_on_method(mocker):
    """Test that method is called while using real implementation."""
    obj = MyClass()
    spy = mocker.spy(obj, 'internal_method')
    
    obj.public_method()
    
    spy.assert_called_once()
```

### Mock Assertions

```python
# Verify call count
mock_func.assert_called_once()
mock_func.assert_called()
mock_func.assert_not_called()
assert mock_func.call_count == 3

# Verify call arguments
mock_func.assert_called_with(arg1, arg2, key=value)
mock_func.assert_called_once_with(arg1, arg2)
mock_func.assert_any_call(arg1)

# Verify call order
assert mock_func.call_args_list == [
    mocker.call(arg1),
    mocker.call(arg2)
]
```

### When to Mock

- External API calls or network requests
- File system operations (when not testing I/O specifically)
- Database connections
- Expensive computations
- Random number generation (for deterministic tests)
- Time-dependent operations

### When NOT to Mock

- The code under test itself
- Simple data structures
- Pure functions with no side effects
- Testing I/O operations (use `tmp_path` fixture instead)

---

## Assertions

### Assertion Style

Use plain `assert` statements (pytest style), not `unittest` assertions:

```python
# Good (pytest style)
assert result == expected
assert len(data) > 0
assert isinstance(obj, MyClass)
assert value in collection

# Bad (unittest style)
self.assertEqual(result, expected)
self.assertTrue(len(data) > 0)
self.assertIsInstance(obj, MyClass)
```

### Assertion Messages

Add helpful messages for complex assertions:

```python
assert result == expected, f"Expected {expected}, got {result}"
assert all(x > 0 for x in values), "All values must be positive"
```

### NumPy Array Assertions

```python
import numpy as np

# Exact equality
assert np.array_equal(array1, array2)

# Approximate equality
assert np.allclose(array1, array2, rtol=1e-5, atol=1e-8)

# Shape checks
assert array.shape == (10, 10)

# Value checks
assert np.all(array > 0)
assert np.any(np.isnan(array))
```

### Exception Testing

```python
def test_raises_value_error(self):
    """Test that invalid input raises ValueError."""
    with pytest.raises(ValueError, match="must be positive"):
        process_data(-1)

def test_raises_any_exception(self):
    """Test that operation raises any exception."""
    with pytest.raises(Exception):
        risky_operation()
```

---

## Naming Conventions

### Test Files

- `test_<module_name>.py`
- Examples: `test_ivim_model.py`, `test_parameters.py`, `test_hdf5.py`

### Test Classes

- `Test<ClassName>` for testing a specific class
- `Test<Functionality>` for testing a feature
- Examples: `TestIVIMModel`, `TestParameterLoading`, `TestBoundaryValidation`

### Test Functions/Methods

Pattern: `test_<what>_<condition>_<expected>`

```python
# Good examples
def test_fit_converges_with_good_initial_values(self):
def test_load_parameters_raises_error_for_invalid_file(self):
def test_model_returns_correct_shape(self):

# Bad examples
def test_fit(self):  # Too vague
def test_1(self):    # Meaningless
def test_the_fitting_procedure_works_correctly_when_given_good_data(self):  # Too long
```

### Fixture Names

- Lowercase with underscores
- Descriptive of what they provide
- Include module/type prefix for clarity

```python
ivim_params
nnls_result
biexp_fitdata
sample_nifti_file
temp_output_path
```

---

## Docstrings

### Requirement

**Every test must have a docstring** explaining:
1. What is being tested
2. Why it matters (if not obvious)
3. Expected behavior

### Format

Use single-line docstrings for simple tests:

```python
def test_model_initialization(self):
    """Test that IVIM model initializes with valid parameters."""
    pass
```

Use multi-line docstrings for complex tests:

```python
def test_fit_with_segmented_approach(self, fitdata, params):
    """
    Test IVIM fitting using segmented approach.
    
    The segmented approach first fits high b-values for diffusion,
    then fits low b-values for perfusion parameters. This test
    verifies that the two-stage fitting converges correctly.
    """
    pass
```

### Class Docstrings

```python
class TestIVIMFitting:
    """
    Test suite for IVIM model fitting functionality.
    
    Tests cover various fitting scenarios including:
    - Different initial parameter values
    - Segmented vs. simultaneous fitting
    - GPU vs. CPU execution
    - Boundary condition handling
    """
    pass
```

---

## Parametrization

### Use `@pytest.mark.parametrize`

Parametrize tests instead of writing repetitive test functions:

```python
@pytest.mark.parametrize("b_value,expected", [
    (0, 1.0),
    (100, 0.9),
    (1000, 0.3),
])
def test_signal_decay(self, b_value, expected):
    """Test signal decay at different b-values."""
    signal = compute_signal(b_value)
    assert np.isclose(signal, expected, rtol=0.1)
```

### Multiple Parameters

```python
@pytest.mark.parametrize("model_type", ["monoexp", "biexp", "triexp"])
@pytest.mark.parametrize("noise_level", [0.0, 0.01, 0.05])
def test_fitting_robustness(self, model_type, noise_level):
    """Test fitting robustness across models and noise levels."""
    pass
```

### Parametrize with Fixtures

```python
@pytest.mark.parametrize("params_file", [
    pytest.lazy_fixture("biexp_params"),
    pytest.lazy_fixture("triexp_params"),
])
def test_load_from_different_formats(self, params_file):
    """Test parameter loading from various file formats."""
    pass
```

### IDs for Readability

```python
@pytest.mark.parametrize("value,expected", [
    (0, "zero"),
    (1, "one"),
    (2, "two"),
], ids=["zero", "one", "two"])
def test_number_to_string(self, value, expected):
    """Test number to string conversion."""
    pass
```

---

## Test Markers

### Standard Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.slow
def test_large_dataset_processing(self):
    """Test processing of large dataset (may take several minutes)."""
    pass

@pytest.mark.gpu
def test_gpu_acceleration(self):
    """Test that GPU acceleration works correctly."""
    pass

@pytest.mark.integration
def test_full_pipeline(self):
    """Test complete processing pipeline end-to-end."""
    pass
```

### Custom Markers

Define custom markers in `pytest.ini` or `conftest.py`:

```python
# conftest.py
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "integration: mark test as integration test")
```

### Skip/XFail Markers

```python
@pytest.mark.skip(reason="Feature not yet implemented")
def test_future_feature(self):
    """Test for feature planned in next release."""
    pass

@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on Windows")
def test_unix_specific_feature(self):
    """Test Unix-specific functionality."""
    pass

@pytest.mark.xfail(reason="Known bug #123")
def test_known_issue(self):
    """Test that currently fails due to known bug."""
    pass
```

### Running Specific Markers

```bash
# Run only GPU tests
pytest -m gpu

# Run all except slow tests
pytest -m "not slow"

# Run GPU or integration tests
pytest -m "gpu or integration"
```

---

## Best Practices

### 1. Test Independence

Each test should be independent and not rely on other tests:

```python
# Good: Each test is self-contained
class TestDataProcessing:
    def test_load_data(self, data_file):
        data = load_data(data_file)
        assert data is not None
    
    def test_process_data(self, data_file):
        data = load_data(data_file)
        result = process_data(data)
        assert result.shape == data.shape

# Bad: Tests depend on each other
class TestDataProcessing:
    def test_load_data(self):
        self.data = load_data("file.txt")
        assert self.data is not None
    
    def test_process_data(self):
        result = process_data(self.data)  # Depends on previous test!
        assert result is not None
```

### 2. One Assertion Per Test (Guideline)

Prefer focused tests with single logical assertions:

```python
# Good: Focused tests
def test_result_has_correct_shape(self, result):
    """Test that result array has expected shape."""
    assert result.shape == (10, 10)

def test_result_values_positive(self, result):
    """Test that all result values are positive."""
    assert np.all(result > 0)

# Acceptable: Related assertions
def test_result_properties(self, result):
    """Test that result has correct shape and positive values."""
    assert result.shape == (10, 10)
    assert np.all(result > 0)
```

### 3. Use Fixtures for Setup

Avoid repetitive setup code in tests:

```python
# Good: Use fixture
@pytest.fixture
def fitted_model(ivim_params, fitdata):
    """Fitted IVIM model for testing."""
    model = IVIMModel(ivim_params)
    model.fit(fitdata)
    return model

def test_model_coefficients(fitted_model):
    assert len(fitted_model.coefficients) == 3

# Bad: Repeat setup in every test
def test_model_coefficients(ivim_params, fitdata):
    model = IVIMModel(ivim_params)
    model.fit(fitdata)
    assert len(model.coefficients) == 3
```

### 4. Test Edge Cases

Always test boundary conditions and edge cases:

```python
def test_empty_input(self):
    """Test behavior with empty input."""
    result = process_data([])
    assert result == []

def test_single_value(self):
    """Test behavior with single value."""
    result = process_data([42])
    assert result == [42]

def test_negative_values(self):
    """Test that negative values raise error."""
    with pytest.raises(ValueError):
        process_data([-1, -2, -3])
```

### 5. Use Descriptive Variable Names

```python
# Good
def test_biexp_model_evaluation(self):
    b_values = np.array([0, 50, 100, 500, 1000])
    params = [0.8, 0.2, 0.003]
    expected_signal = np.array([1.0, 0.95, 0.85, 0.45, 0.25])
    
    actual_signal = evaluate_model(b_values, params)
    assert np.allclose(actual_signal, expected_signal, rtol=0.1)

# Bad
def test_biexp_model_evaluation(self):
    x = np.array([0, 50, 100, 500, 1000])
    p = [0.8, 0.2, 0.003]
    e = np.array([1.0, 0.95, 0.85, 0.45, 0.25])
    
    a = evaluate_model(x, p)
    assert np.allclose(a, e, rtol=0.1)
```

### 6. Avoid Test Logic

Tests should be simple and straightforward:

```python
# Good: Simple, clear test
def test_sum_calculation(self):
    result = calculate_sum([1, 2, 3, 4])
    assert result == 10

# Bad: Complex logic in test
def test_sum_calculation(self):
    values = [1, 2, 3, 4]
    expected = sum(values)  # Don't recalculate in test!
    result = calculate_sum(values)
    assert result == expected
```

### 7. Fast Tests

Keep tests fast where possible:
- Mock expensive operations
- Use small datasets for testing logic
- Use `tmp_path` instead of real filesystem when possible
- Mark slow tests with `@pytest.mark.slow`

```python
@pytest.mark.slow
def test_large_scale_processing(self):
    """Test processing of production-scale dataset."""
    # Process 10GB of data
    pass

def test_processing_logic(self):
    """Test processing logic with small sample."""
    # Process 10 samples to verify logic
    pass
```

### 8. Temporary Files

Use pytest's `tmp_path` fixture for file operations:

```python
def test_save_and_load(tmp_path):
    """Test saving and loading data from file."""
    output_file = tmp_path / "output.json"
    
    data = {"key": "value"}
    save_data(data, output_file)
    
    loaded = load_data(output_file)
    assert loaded == data
    # No cleanup needed - tmp_path is automatically removed
```

---

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run specific file
pytest tests/test_ivim_model.py

# Run specific test
pytest tests/test_ivim_model.py::TestIVIMModel::test_initialization

# Run tests matching pattern
pytest -k "model"

# Run with markers
pytest -m gpu
pytest -m "not slow"

# Show print statements
pytest -s

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Run failed tests from last run
pytest --lf
```

### Coverage

```bash
# Run with coverage
pytest --cov=pyneapple --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Parallel Execution

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest -n auto
```

---

## LLM Prompting Guidelines

When using LLMs to generate tests, provide these instructions:

### Example Prompt Template

```
Create pytest tests for [MODULE_NAME] following these requirements:

1. Use class-based test organization (Test[ClassName])
2. Use pytest-mock (mocker fixture) for all mocking, NOT unittest.mock
3. Add docstrings to all test functions explaining what is tested
4. Use parametrize for testing multiple scenarios
5. Follow naming convention: test_<action>_<condition>_<expected>
6. Use plain assert statements (not unittest assertions)
7. Make tests independent - no shared state between tests
8. Include edge case testing (empty inputs, boundary values, errors)
9. Use appropriate fixtures from conftest.py: [LIST RELEVANT FIXTURES]
10. Add appropriate markers: @pytest.mark.slow, @pytest.mark.gpu, etc.

Example structure:
```python
class TestMyClass:
    """Test suite for MyClass functionality."""
    
    def test_method_returns_expected_value(self, fixture_name):
        """Test that method returns correct value with valid input."""
        # Arrange
        obj = MyClass()
        
        # Act
        result = obj.method(input_value)
        
        # Assert
        assert result == expected_value
```
```

---

## Questions or Updates?

If you have questions about these guidelines or suggestions for improvements, please:
1. Open an issue in the repository
2. Discuss in team meetings
3. Submit a pull request with proposed changes

Keep these guidelines up-to-date as the project evolves!
