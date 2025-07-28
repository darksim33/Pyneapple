# TOML Parameter Files in Pyneapple

This document describes how to use TOML format for parameter files in Pyneapple.

## Overview

Starting with version 1.6.1, Pyneapple supports TOML (Tom's Obvious, Minimal Language) format for parameter files in addition to the existing JSON format. TOML provides a more readable and maintainable configuration format with better support for comments and hierarchical data structures.

## Features

- Load parameter files in TOML format
- Save parameter files to TOML format
- Automatic detection of file format based on file extension
- Full compatibility with existing parameter classes
- Support for comments in parameter files

## Why Use TOML?

TOML offers several advantages over JSON for configuration files:

1. **Comments**: TOML supports comments, making it easier to document your parameter files
2. **Readability**: TOML's syntax is designed to be more human-readable
3. **Hierarchical Structure**: TOML provides a cleaner syntax for nested data structures
4. **Date/Time Support**: TOML has native support for date and time values

## Usage

### Loading Parameters from TOML Files

You can load parameters from a TOML file in the same way you load them from JSON files:

```python
from pyneapple.parameters.ivim import IVIMParams

# Load parameters from a TOML file
params = IVIMParams("path/to/parameters.toml")
```

The file format is automatically detected based on the file extension.

### Loading Explicitly from TOML

You can also explicitly load parameters from a TOML file using the `load_from_toml` method:

```python
from pyneapple.parameters.ivim import IVIMParams

# Create parameters object
params = IVIMParams()

# Explicitly load from TOML
params.load_from_toml("path/to/parameters.toml")
```

### Saving Parameters to TOML Files

You can save parameters to a TOML file using the `save_to_toml` method:

```python
from pyneapple.parameters.ivim import IVIMParams

# Create or load parameters
params = IVIMParams("path/to/parameters.json")

# Modify parameters if needed
params.fit_type = "gpu"
params.max_iter = 200

# Save to TOML file
params.save_to_toml("path/to/parameters.toml")
```

## TOML Parameter File Example

Here's an example of a parameter file in TOML format for IVIM fitting:

```toml
[General]
# IVIM Parameter File
Class = "IVIMParams"

# Basic parameters
model = "bi-exponential"
fit_type = "multi"
n_pools = 4
max_iter = 250
fit_tolerance = 1e-6
fit_reduced = false
fit_t1 = false

[Model]
model = "biexp"  
fit_reduced = false  
fit_t1 = true 
fit_S0 = false
mixing_time = 20 

# Boundaries section
[boundaries]
# Define D (diffusion) parameter boundaries
[boundaries.D]
# Format: [initial_value, lower_bound, upper_bound]
"0" = [1.0, 0.1, 3.0]   # D_slow component
"1" = [10.0, 5.0, 50.0] # D_fast component

# Define f (fraction) parameter boundaries
[boundaries.f]
"0" = [0.7, 0.1, 0.9] # f_slow component
"1" = [0.3, 0.1, 0.9] # f_fast component

# Define S0 (signal) parameter boundaries
[boundaries.S]
"0" = [1.0, 0.5, 1.5] # S0 parameter

# Optional T1 parameter boundaries
[boundaries.T1]
"0" = [1000.0, 500.0, 2000.0] # T1 parameter in ms

# B-values array
b_values = [0, 50, 100, 150, 200, 400, 600, 800]
```

## Dependencies

The TOML parameter support requires the following packages:

- `tomli` for Python < 3.11 (Python 3.11+ has built-in `tomllib`)
- `tomli-w` for writing TOML files (required only when saving parameters to TOML format)

These dependencies are automatically installed with Pyneapple.

## Converting Between Formats

You can easily convert between JSON and TOML formats:

```python
from pyneapple.parameters.ivim import IVIMParams

# Load from JSON
params = IVIMParams("parameters.json")

# Save to TOML
params.save_to_toml("parameters.toml")

# Or load from TOML
params = IVIMParams("parameters.toml")

# Save to JSON
params.save_to_json("parameters.json")
```

## Limitations

- TOML doesn't support some specialized JSON features like arbitrary precision numbers
- TOML serialization requires the `tomli-w` package to be installed
