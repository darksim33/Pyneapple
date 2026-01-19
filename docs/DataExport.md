# Exporting Fit Results

Pyneapple provides multiple export formats for fit results to suit different analysis workflows:
- **HDF5** (recommended) - Full data preservation
- **NIfTI** - Medical imaging standard
- **Excel** - Tabular data for spreadsheet analysis

---

## HDF5 Export (Recommended)

The recommended format is **HDF5**. The data is stored in the same way it is handled internally, preserving all information from the fitting process.

### Data Structure

HDF5 files store results as dictionaries with pixel coordinates $(x,y,z)$ or segmentation indices $(n)$ as keys:

- **D**: Diffusion coefficients as arrays $(D_1 ... D_n)$
- **f**: Fractions for each coefficient as arrays $(f_1 ... f_n)$
- **S0**: Absolute signal intensity
- **raw**: Raw decay signal used to calculate fit results
- **curve**: Decay curve calculated from the fit results
- **spectrum**: Diffusion spectrum for coefficients and fractions

Numpy arrays are efficiently stored using the [*sparse*](https://github.com/pydata/sparse/) package.


### Usage

```python
from pyneapple import IVIMResults
from radimgarray import RadImgArray

# Save as dictionary (preserves exact internal structure)
results.save_to_hdf5('results.h5')

# Save as array (converts to spatial arrays)

# Load your image
img = RadImgArray('path/to/image.nii.gz')
results.save_to_hdf5_as_array('results_array.h5', img=img)
```
When to Use HDF5

- **✓** Full analysis pipeline with Python
- **✓** Need to preserve all fitted data
- **✓** Working with large datasets
- **✓** Need to reload results for further processing

## NIfTI Export

Export results as NIfTI files for use with medical imaging software (FSL, AFNI, ITK-SNAP, 3D Slicer, etc.).

### File Naming Convention

NIfTI files are automatically named based on the parameter:

**Combined mode** (default):
- `{basename}_d.nii.gz` - All diffusion coefficients (4D: x, y, z, components)
- `{basename}_f.nii.gz` - All fractions (4D: x, y, z, components)
- `{basename}_S0.nii.gz` - Signal intensity (3D)
- `{basename}_t1.nii.gz` - T1 relaxation time (IVIM only, if fitted)

**Separate files mode**:
- `{basename}_d_0.nii.gz`, `{basename}_d_1.nii.gz`, ... - Individual D values (3D)
- `{basename}_f_0.nii.gz`, `{basename}_f_1.nii.gz`, ... - Individual fractions (3D)
- `{basename}_S0.nii.gz` - Signal intensity (3D)
- `{basename}_t1.nii.gz` - T1 relaxation time (IVIM only, if fitted)

### Basic Usage

```python
from pyneapple import IVIMResults
from radimgarray import RadImgArray

# Load your image
img = RadImgArray('path/to/image.nii.gz')

# Perform fitting and get results
# ... fitting code ...

# Save as combined 4D files (default)
results.save_to_nii('output_path/results', img, dtype=float)

# Save each component as separate 3D files
results.save_to_nii('output_path/results', img, dtype=float, separate_files=True)
```

### Advanced Options

#### Data Type Selection

Control the precision and file size with the `dtype` parameter:

```python
# High precision (larger files)
results.save_to_nii('results', img, dtype=float)  # float64
results.save_to_nii('results', img, dtype=np.float32)

# Integer (smaller files, loses decimal precision)
results.save_to_nii('results', img, dtype=int)

# Default behavior
results.save_to_nii('results', img, dtype=None)
```

#### NNLS-Specific: Cutoff Ranges

NNLS results have variable numbers of compartments per pixel. Apply cutoffs to normalize:

```python
from pyneapple import NNLSResults

# Option 1: Apply cutoffs before saving
cutoffs = [(0.0, 0.001), (0.001, 0.01), (0.01, 0.1)]  # Slow, intermediate, fast
results.apply_cutoffs(cutoffs)
results.save_to_nii('results', img, dtype=float)

# Option 2: Pass cutoffs during save
results.save_to_nii('results', img, dtype=float, cutoffs=cutoffs)
```
Cutoff ranges explanation:**
- `(0.0, 0.001)`: Slow diffusion (restricted)
- `(0.001, 0.01)`: Intermediate diffusion
- `(0.01, 0.1)`: Fast diffusion (perfusion-like)

After cutoffs, each pixel has the same number of compartments (some may be NaN if no peaks found in that range).

When to Use NIfTI

- **✓** Visualization in medical imaging software
- **✓** Registration with anatomical images
- **✓** Integration with FSL/AFNI/SPM pipelines
- **✓** Creating region-of-interest maps
- **✓** Sharing results with collaborators using standard tools

## Excel Export

Export results as Excel spreadsheets for tabular analysis, statistics, and plotting.

### File Structure

Excel files organize data in rows (pixels/segments) and columns (parameters):

|   | pixel/seg | parameter | value |
|---|-----------|-----------|-------|
| 0 | (0,0,0)   | D_0       | 0.001 |
| 1 | (0,0,0)   | D_1       | 0.015 |
| 2 | (0,0,0)   | f_0       | 0.7   |
| 3 | (0,0,0)   | f_1       | 0.3   |
| 4 | (0,0,0)   | S0        | 1000  |

### Basic Usage

```python
# Pixel-wise results
results.save_to_excel('results.xlsx')

# Segmentation-wise results
results.save_to_excel('seg_results.xlsx', is_segmentation=True)

# Split pixel coordinates into separate columns
results.save_to_excel('results.xlsx', split_index=True)

```

### Output Formats

Pixel-wise (default)

```python
results.save_to_excel('pixel_results.xlsx', split_index=False)
```

|   | pixel     | parameter | value |
|---|-----------|-----------|-------|
| 0 | (0,0,0)   | D_0       | 0.001 |
| 1 | (0,0,0)   | f_0       | 0.7   |
| 2 | (1,1,0)   | D_0       | 0.002 |

#### Pixel-wise with split coordinates

```python
results.save_to_excel('pixel_results.xlsx', split_index=True)
```

|   | x | y | z | parameter | value |
|---|---|---|---|-----------|-------|
| 0 | 0 | 0 | 0 | D_0       | 0.001 |
| 1 | 0 | 0 | 0 | f_0       | 0.7   |
| 2 | 1 | 1 | 0 | D_0       | 0.002 |

#### Segmentation-wise

```python
results.save_to_excel('seg_results.xlsx', is_segmentation=True)
```

|   | seg_number | parameter | value |
|---|------------|-----------|-------|
| 0 | 1          | D_0       | 0.001 |
| 1 | 1          | f_0       | 0.75  |
| 2 | 2          | D_0       | 0.003 |

### Specialized Excel Exports

#### Diffusion Spectrum

Export the full diffusion spectrum for NNLS or IVIM results:

```python
# NNLS automatically uses correct bins
results.save_spectrum_to_excel('spectrum.xlsx')

# IVIM with custom bins
bins = results._get_bins(101, (0.0007, 0.3))
results.save_spectrum_to_excel('spectrum.xlsx', bins=bins)

# Segmentation-wise spectrum
results.save_spectrum_to_excel('spectrum.xlsx', is_segmentation=True)
```

**Output format:**

|   |  pixel  | 0.0007 | 0.0008 | 0.0009 | ... | 0.3   |
|---|---------|--------|--------|--------|-----|-------|
| 0 | (0,0,0) | 0.0    | 0.3    | 0.7    | ... | 0.0   |
| 1 | (1,1,0) | 0.0    | 0.0    | 0.5    | ... | 0.1   |

#### Fitted Decay Curve

Export the fitted decay curves:

```python
b_values = [0, 10, 20, 30, 50, 100, 150, 200, 400, 800]

results.save_fit_curve_to_excel('curves.xlsx', b_values=b_values)

# With split index
results.save_fit_curve_to_excel('curves.xlsx', b_values=b_values, split_index=True)
```

**Output format:**

|   |  pixel  | 0 | 10 | 20 | 30 | ... | 800 |
|---|---------|---|----|----|----|----|-----|
| 0 | (0,0,0) | 1000 | 950 | 900 | 850 | ... | 100 |
| 1 | (1,1,0) | 1100 | 1040 | 980 | 920 | ... | 110 |

### Complete Example

```python
from pyneapple import NNLSResults
from pathlib import Path

# After fitting...
output = Path('results')

# Standard results
results.save_to_excel(output / 'fit_results.xlsx', split_index=True)

# Spectrum with bins
results.save_spectrum_to_excel(output / 'spectrum.xlsx')

# Fitted curves
b_values = params.b_values
results.save_fit_curve_to_excel(output / 'curves.xlsx', b_values=b_values)

# Segmentation summary
results.save_to_excel(output / 'segment_summary.xlsx', is_segmentation=True)
```

### When to Use Excel

- **✓** Statistical analysis in Excel, R, or Python pandas
- **✓** Creating custom plots and visualizations
- **✓** Sharing tabular results with non-programmers
- **✓** Quick data inspection and exploration
- **✓** Merging with other tabular data sources

---

## Format Comparison

| Feature | HDF5 | NIfTI | Excel |
|---------|------|-------|-------|
| Full data preservation | ✓ | ✗ | ✗ |
| Spatial visualization | ✗ | ✓ | ✗ |
| Spreadsheet analysis | ✗ | ✗ | ✓ |
| File size (large data) | Small | Medium | Large |
| Medical imaging tools | ✗ | ✓ | ✗ |
| Python reloading | ✓ | Partial | Partial |
| Human readable | ✗ | ✗ | ✓ |

---

## Best Practices

### 1. Save Multiple Formats

```python
# Preserve everything in HDF5
results.save_to_hdf5('results.h5')

# Create NIfTI for visualization
results.save_to_nii('results_nii', img, dtype=np.float32)

# Export to Excel for statistics
results.save_to_excel('results.xlsx', split_index=True)
```

### 2. Organize Output

```python
from pathlib import Path
import datetime

# Create timestamped output directory
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output = Path(f'results_{timestamp}')
output.mkdir(exist_ok=True)

# Save all formats
results.save_to_hdf5(output / 'data.h5')
results.save_to_nii(output / 'nifti' / 'maps', img, dtype=np.float32)
results.save_to_excel(output / 'tables' / 'summary.xlsx')
```

### 3. Use Appropriate Data Types

```python
# For visualization (balance quality and size)
results.save_to_nii('visual', img, dtype=np.float32)

# For precise calculations (larger files)
results.save_to_nii('precise', img, dtype=np.float64)

# For masks or integer data
results.save_to_nii('mask', img, dtype=int)
```

### 4. Document Your Exports

```python
# Save parameter file alongside results
import shutil
shutil.copy('params.toml', output / 'params.toml')

# Or save metadata
metadata = {
    'date': datetime.datetime.now().isoformat(),
    'params': params.__dict__,
    'n_pixels': len(results.D),
}
import json
with open(output / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

---

## Troubleshooting

### NIfTI files are too large

Use float32 instead of float64:
```python
results.save_to_nii('results', img, dtype=np.float32)
```

### NNLS results fail to save as NIfTI

Apply cutoffs first:
```python
results.apply_cutoffs([(0.0, 0.001), (0.001, 0.01), (0.01, 0.1)])
results.save_to_nii('results', img, dtype=float)
```

### Excel file is too large

Consider using HDF5 or NIfTI for large datasets, or save only segmentation-wise results:
```python
results.save_to_excel('summary.xlsx', is_segmentation=True)
```

### Need to reload results

Use HDF5 format:
```python
# Save
results.save_to_hdf5('results.h5')

# Load
from pyneapple.io.hdf5 import load_from_hdf5
data = load_from_hdf5('results.h5')
results.load_from_dict(data)
```
