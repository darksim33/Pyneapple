import os.path
import sys

from pathlib import Path
from multiprocessing import freeze_support
from glob import glob
from tqdm import tqdm

from pyneapple.utils.nifti import Nii, NiiSeg
from pyneapple.fit.fit import FitData

if __name__ == "__main__":
    """
    Script to fit any number of images using multiple segmentations and different fitting techniques, the latter being
    specified in the 'fitting_models' variable. Matching fitting image and segmentation file names must be the same
    string with an additional suffix for the segmentation. Fitting parameters may be specified in 'fitting_parameters'.
    All files (images and segmentations) need to be located inside the 'folder_path'.

    Executing this script in VSC will produce beautiful progress bars as output. PyCharm does not support this feature
    properly.

    Attributes
    ----------
    folder_path : str
        Set home folder containing images (.nii) and ROIs (.nii.gz) to be fitted.

    fitting_models : list[str]
        Specify fitting procedures to be carried out.

    fitting_parameters : list[Path]
        List of json file paths containing parameters for each fitting model. Needs to be sorted in same order as
        'fitting_models'.
    """

    # Initialisation
    freeze_support()

    folder_path = r"data/MEDIA_data/"
    fitting_models = ["NNLS", "NNLSCV"]
    fitting_parameters = [
        Path(r"data/MEDIA_data/params/default_params_NNLS.json"),
        Path(r"data/MEDIA_data/params/default_params_NNLSCV.json"),
        # Path(r"data/MEDIA_data/params/default_params_IVIM_tri.json"),
        # Path(r"data/MEDIA_data/params/default_params_IDEAL_tri.json"),
    ]

    # Filter path for img (.nii) and seg files (.nii.gz)
    for img, seg in [(glob(folder_path + "*.nii"), glob(folder_path + "*.nii.gz"))]:
        img_files = img
        seg_files = seg

    # Mute all print outputs for better display of progress bars | optional
    sys.stdout = open(os.devnull, "w")

    # Fitting each image
    for img_file in tqdm(img_files, desc=" image", position=0):
        # To all associated segmentations (based on matching filename)
        seg_files_subject = [seg for seg in seg_files if Path(img_file).stem in seg]

        for seg_file in tqdm(
            seg_files_subject,
            desc=" segmentation",
            position=1,
            leave=False,
        ):
            # Using the specified fitting models
            for model, fitting_model in enumerate(
                tqdm(
                    fitting_models,
                    desc=" fitting model",
                    position=2,
                    leave=False,
                )
            ):
                # Initiate fitting procedure
                data = FitData(
                    model=fitting_model,
                    img=Nii(img_file),
                    seg=NiiSeg(seg_file),
                )
                data.fit_params.load_from_json(fitting_parameters[model])

                out_path = str(
                    Path(
                        os.path.dirname(img_file)
                        + "\\"
                        + Path(Path(seg_file).stem).stem
                        + "_"
                        + data.model_name
                    )
                )

                # Fit pixel-wise
                data.fit_pixel_wise(multi_threading=True)
                data.fit_results.save_results_to_excel(Path(out_path + "_pixel.xlsx"))

                if fitting_model == "NNLS":
                    d_AUC, f_AUC = data.fit_params.apply_AUC_to_results(
                        data.fit_results
                    )
                    data.fit_results.save_results_to_excel(
                        Path(out_path + "_pixel_AUC.xlsx"),
                        d_AUC,
                        f_AUC,
                    )

                # Fit segmentation-wise
                data.fit_segmentation_wise()
                data.fit_results.save_results_to_excel(
                    Path(out_path + "_segmentation.xlsx")
                )

                if fitting_model == "NNLS":
                    d_AUC, f_AUC = data.fit_params.apply_AUC_to_results(
                        data.fit_results
                    )
                    data.fit_results.save_results_to_excel(
                        Path(out_path + "_segmentation_AUC.xlsx"),
                        d_AUC,
                        f_AUC,
                    )

                    # # Create heatmaps
                    # fit_data.fit_results.create_heatmap(
                    #     fit_data,
                    #     Path(
                    #         os.path.dirname(img_file)
                    #         + "\\"
                    #         + Path(img_file).stem
                    #         + "_"
                    #         + fit_data.model_name
                    #         + "_heatmaps"
                    #     ),
                    #     seg.slices_contain_seg,
                    # )

    print("Done")
