# Pyneapple 🍍 
<img src=".github/logo.png" alt="logo" style="width:128px;height:128px;"/> 

## Why Pyneapple?

Pyneapple is an advanced tool for analysing multi-exponential signal data in MR DWI images. It is able to apply a
variety of different fitting algorithms (NLLS, NNLS, ...) to the measured diffusion data and to compare
multi-exponential fitting methods. Thereby it can determine the total number of components contributing to the
corresponding multi-exponential signal fitting approaches and analyses the results by calculating the corresponding
diffusion parameters. Fitting can be customised to be performed on a pixel by pixel or segmentation-wise basis.

## Installation 

There are different ways to get Pyneapple running depending on the desired use case. If you want to integrate Pyneapple
in your existing workflow to use the processing utilities the best way to go is using *pip* with the *git* tag.

```console
pip install git+https://github.com/darksim33/Pyneapple
```

If your planing on altering the code by forking or cloning the repository, Pyneapple is capable of using [
*poetry*](https://python-poetry.org). There are different ways to install *poetry*. For Windows and Linux a straight
forward approach is using [*pipx*](https://pipx.pypa.io/stable/installation/). First you need to install *pipx* using
*pip* which basically follows the same syntax as pip itself. Afterward you can install poetry in an isolated environment
created by *pipx*.

```console
python -m pip install pipx
python -m pipx install poetry
```

To use an editable installation of Pyneapple navigate to the repository directory make sure all submodules are 
initialized properly and perform the installation using the local virtual environment. 

```console
cd <path_to_the_repository>
git submodule update --init --recursive
poetry install
```

There are tow additional options for _development_. If you want to take advantage of the
testing framework you can install the required dependencies by:

```console
poetry install --with dev
```

## Testing

Pyneapple includes a comprehensive test suite to ensure code quality and reliability. The tests are written using pytest and follow established guidelines for consistency and maintainability.

### Running Tests

Run all tests:
```console
pytest
```

Run tests with verbose output:
```console
pytest -v
```

Run specific test file:
```console
pytest tests/test_ivim_model.py
```

Run tests with specific marker:
```console
pytest -m gpu          # Run only GPU tests
pytest -m "not slow"   # Skip slow tests
```

Run tests with coverage:
```console
pytest --cov=pyneapple --cov-report=html
```

### Test Guidelines

When writing tests or using LLMs to generate tests, please follow the guidelines documented in [TestingGuidelines.md](./docs/TestingGuidelines.md). Key points include:

- Use class-based test organization for related test cases
- Use pytest-mock (mocker fixture) for all mocking, not unittest.mock
- Add docstrings to all test functions
- Use parametrize for testing multiple scenarios
- Follow naming convention: `test_<action>_<condition>_<expected>`
- Make tests independent with no shared state

For complete details, see [Testing Guidelines](./docs/TestingGuidelines.md).

## Documentation

 - Fitting
    - [Genral Instructions](./docs/Fitting.md): General instruction to explain the different steps
    - [Fitting Parameters](docs/Parameters.md): Detailed Parameter explanation
    - [Fit Models](./docs/FitModels.md): Description of different available models
    - [Examples](./docs/FitExamples.md): Collection of different examples to perform basic fittings.
 - Testing
    - [Testing Guidelines](./docs/TestingGuidelines.md): Comprehensive guidelines for writing and maintaining tests

___