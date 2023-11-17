from src.fit.fit import FitData
import pandas as pd


def save_results(data: FitData):
    result_dict = set_up_results_struct(data)

    result_dict = pd.DataFrame(result_dict).T
    result_dict.to_excel(f"PyNeapple_results_{data.model_name}.xlsx")


def set_up_results_struct(data: FitData):
    result_dict = {}
    index = 0
    current_pixel = 0

    for idx in data.fit_results.d.keys():
        n_comps = len(data.fit_results.d[idx])
        current_pixel += 1

        for comp in range(0, n_comps):
            # Implement saving of multiple methods into one file? May be more convenient,
            # but single files are clearer, don't need caching and no further coding needed

            index += 1
            result_dict[index] = {
                # 'ROI': 0,
                "pixel": current_pixel,
                "pixel_position": idx,
                "method": data.model_name,
                "compartment": comp + 1,
                "D": data.fit_results.d[idx][comp],
                "f": data.fit_results.f[idx][comp],
                "foundCompartments": n_comps,
            }

    return result_dict
