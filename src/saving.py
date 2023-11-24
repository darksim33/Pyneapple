from src.fit.fit import FitData
import pandas as pd
from pathlib import Path
from src.utils import Nii


def save_results(data: FitData):
    result_dict = set_up_results_struct(data)

    # Save results for D and f to Excel sheet
    result_dict = pd.DataFrame(result_dict).T
    result_dict.to_excel(Path(f"data/results/PyNeapple_results_{data.model_name}.xlsx"))

    # Save spectrum as Nii
    spec = Nii().from_array(data.fit_results.spectrum)
    spec.save(Path(r"data/results/spec.nii"))


def set_up_results_struct(data: FitData):
    result_dict = {}
    index = 0
    current_pixel = 0

    for key, d_values in data.fit_results.d.items():
        n_comps = len(d_values)
        current_pixel += 1

        for comp, d_comp in enumerate(d_values):
            # Implement saving of multiple methods into one file? May be more convenient,
            # but single files are clearer, don't need caching and no further coding needed

            index += 1
            result_dict[key + (comp + 1,)] = {
                "index": index,
                "element": current_pixel,
                "pixel_position": key,  # Only available for pixel-wise fitting?
                "method": data.model_name,
                "D": d_comp,
                "f": data.fit_results.f[key][comp],
                "found_compartments": n_comps,
            }

    return result_dict
