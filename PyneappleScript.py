from pathlib import Path
from multiprocessing import freeze_support
from src.utils import Nii, NiiSeg
from src.fit.fit import FitData

from src.saving import save_results
from src.plotting import create_heatmaps

if __name__ == "__main__":
    freeze_support()

    img_files = [Path(r"data/test_img_176_176.nii")]
    seg_files = [Path(r"data/test_mask.nii.gz")]
    out_files = [Path(r"data/results/spec.nii")]

    for img_file, seg_file, out_file in zip(img_files, seg_files, out_files):
        img = Nii(img_file)
        seg = NiiSeg(seg_file)

        fit_data = FitData("multiExp", img, seg)
        fit_data.fit_params.load_from_json(Path(r"data/test_params_nnls.json"))

        fit_data.fit_pixel_wise(multi_threading=False)

        d_AUC, f_AUC = fit_data.fit_params.apply_AUC_to_results(fit_data.fit_results)
        create_heatmaps(fit_data, d_AUC, f_AUC)

        save_results(fit_data)

    print("Done")
