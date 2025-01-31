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

If your locked behind a proxy server you might need to commend the dependencies in the "[tool.poetry.dependencies]"
section of the [_pyproject.toml_](pyproject.toml) (except the recommended python version which is mandatory).
Thereafter, you need to install the virtual environment and the packages manually.

There are tow additional options for _development_. If you want to take advantage of the
testing framework you can install the required dependencies by:

```console
poetry install --with dev
```
## Documentation

 - [Fitting Instructions](./docs/Fitting.md)
 - [Fitting Parameters](docs/Parameters.md)
 - [Model Class](docs/ModelClass.md)

___