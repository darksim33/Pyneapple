import os.path

from pathlib import Path
from multiprocessing import freeze_support

from src.utils import Nii, NiiSeg
from src.fit.fit import FitData

if __name__ == "__main__":
    freeze_support()

    img_files = [Path(r"data/test_img_176_176.nii")]
    seg_files = [Path(r"data/test_mask.nii.gz")]
    out_files = [Path(r"data/results/spec.nii")]

    for img_file, seg_file, out_file in zip(img_files, seg_files, out_files):
        img = Nii(img_file)
        seg = NiiSeg(seg_file)

        # Initiate fitting procedure
        fit_data = FitData("IVIM", img, seg)
        fit_data.fit_params.load_from_json(Path(r"data/test_params_nnls.json"))

        fit_data.fit_pixel_wise(multi_threading=False)

        # Save results
        fit_data.fit_results.save_results_to_excel(
            Path(
                os.path.dirname(img_file)
                + "\\"
                + Path(img_file).stem
                + "_"
                + fit_data.model_name
                + "_results.xlsx"
            ),
            fit_data.model_name,
        )

        # Create heatmaps
        d_AUC, f_AUC = fit_data.fit_params.apply_AUC_to_results(fit_data.fit_results)
        img_dim = fit_data.img.array.shape[0:3]

        fit_data.fit_results.create_heatmap(
            img_dim,
            fit_data.model_name,
            d_AUC,
            f_AUC,
            Path(
                os.path.dirname(img_file)
                + "\\"
                + Path(img_file).stem
                + "_"
                + fit_data.model_name
                + "_heatmaps"
            ),
        )

    print("Done")
