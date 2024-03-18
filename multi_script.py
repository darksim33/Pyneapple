import os.path
import sys

from pathlib import Path
from multiprocessing import freeze_support
from glob import glob
from tqdm import tqdm
from os import devnull

from src.utils import Nii, NiiSeg
from src.fit.fit import FitData


if __name__ == "__main__":
    """
    Script to fit any number of images using multiple segmentations and fitting techniques, the latter being
    specified in the 'fitting_models' variable. All files need to be located inside the 'folder_path'.

    Attributes
    ----------
    folder_path : str
        Set home folder containing images (.nii) and ROIs (.nii.gz) to be fitted.

    fitting_models : str
        Specify fitting procedures to be carried out.

    fitting_parameters : json | None
        Optional json file containing parameters for each fitting model. Needs to be sorted in same order as
        'fitting_models'. If not provided uses standard fitting parameters.
    """

    # Initialisation
    freeze_support()
    folder_path = r"data/MEDIA_data/"
    fitting_models = ["NNLSreg", "IVIM", "IDEAL"]
    fitting_parameters = [
        Path(r"data/MEDIA_data/params/default_params_NNLSreg.json"),
        Path(r"data/MEDIA_data/params/default_params_IVIM_tri.json"),
        Path(r"data/MEDIA_data/params/default_params_IDEAL_tri.json"),
    ]

    # Filter path for img (.nii) and seg files (.nii.gz)
    for img, seg in [
        (
            glob(folder_path + "*.nii"),
            glob(folder_path + "*.nii.gz"),
        )
    ]:
        img_files = img
        seg_files = seg

    # Mute all print outputs | optional
    sys.stdout = open(os.devnull, "w")

    # Fitting all models to every segmentation of each image
    for img_file in tqdm(img_files, desc=" image", position=0):
        # Assign subject and scan specific segmentations based on image filename
        seg_files_subject = [seg for seg in seg_files if Path(img_file).stem in seg]

        for seg_file in tqdm(
            seg_files_subject,
            desc=" segmentation",
            position=1,
            leave=False,
        ):
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
                    model=fitting_model, img=Nii(img_file), seg=NiiSeg(seg_file)
                )

                if fitting_parameters:
                    data.fit_params.load_from_json(fitting_parameters[model])
                else:
                    # TODO: Use standard parameters
                    data.fit_params.load_from_json(
                        Path(r"resources\fitting\default_params_IVIM_tri.json")
                    )

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
                data.fit_pixel_wise(multi_threading=False)
                data.fit_results.save_results_to_excel(Path(out_path + "_pixel.xlsx"))

                if fitting_model == "NNLS":  # For NNLS perform AUC
                    d_AUC, f_AUC = data.fit_params.apply_AUC_to_results(
                        data.fit_results
                    )
                    data.fit_results.save_results_to_excel(
                        Path(out_path + "_pixel_AUC.xlsx"), d_AUC, f_AUC
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
                        Path(out_path + "_segmentation_AUC.xlsx"), d_AUC, f_AUC
                    )

    print("Done")
