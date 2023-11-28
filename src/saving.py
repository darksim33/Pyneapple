from src.fit.fit import FitData
import pandas as pd
from pathlib import Path
from src.utils import Nii


def save_results(data: FitData):
    result_df = pd.DataFrame(set_up_results_struct(data)).T

    # Restructure key index and save results
    result_df.reset_index(
        names=["pixel_x", "pixel_y", "slice", "compartment"], inplace=True
    )
    result_df.to_excel(Path(f"data/results/PyNeapple_results_{data.model_name}.xlsx"))

    # Save spectrum as Nii
    spec = Nii().from_array(data.fit_results.spectrum)
    spec.save(Path(r"data/results/spec.nii"))


def set_up_results_struct(data: FitData):
    result_dict = {}
    current_pixel = 0

    for key, d_values in data.fit_results.d.items():
        n_comps = len(d_values)
        current_pixel += 1

        for comp, d_comp in enumerate(d_values):
            result_dict[key + (comp + 1,)] = {
                "element": current_pixel,
                "method": data.model_name,
                "D": d_comp,
                "f": data.fit_results.f[key][comp],
                "n_compartments": n_comps,
            }

    return result_dict
