import os.path
from pathlib import Path
from multiprocessing import freeze_support
from src.utils import Nii, NiiSeg
from src.fit.fit import FitData
import glob

if __name__ == "__main__":
    """
    Script to fit any number of images (.nii) using multiple segmentations (.nii.gz) and fitting techniques,
    the latter being specified in the 'fitting_models' variable. All files need to be located inside the 'folder_path'.

    Attributes
    ----------
    folder_path : str
        Set home folder containing NifTis and ROIs

    fitting_models : str
        Specify fitting procedures to carry out

    fitting_parameters : json | None
        Load optional parameters for each fitting_model in same order as models. If not provided uses standard
        fitting parameters
    """

    # Initialisation
    freeze_support()
    folder_path = [Path(r"data/MEDIA_data")]
    fitting_models = ["IVIM", "NNLS", "IDEAL"]
    fitting_parameters = None

    # Filter for img (.nii) and seg files (.nii.gz)
    for img, seg in glob.glob(fodler_path + "/*.nii"), glob.glob(
        fodler_path + "/*.nii.gz"
    ):
        img_files = img
        seg_files = seg

    # Fitting all models to every segmentation of each image
    for idx, img_file in enumerate(img_files):
        img = Nii(img_file)

        # Extract subject specific segmentations using subject number in filename
        subject_number = Path(img_file).stem[:2]
        seg_files_subject = [
            seg_sub for seg_sub in seg_files if subject_number in seg_sub
        ]

        for seg_file in seg_files_subject:
            seg = NiiSeg(seg_file)

            for model, fitting_model in enumerate(fitting_models):
                # Initiate fitting procedure
                data = FitData(model=fitting_model, img=img, seg=seg)

                if fitting_parameters:
                    data.fit_params.load_from_json(fitting_parameters[model])

                out_path = Path(
                    os.path.dirname(img_file)
                    + "\\"
                    + Path(seg_file).stem
                    + "_"
                    + data.model_name
                )

                # Fit pixel-wise
                data.fit_pixel_wise(multi_threading=True)
                data.fit_results.save_results_to_excel(Path(out_path + "_pixel.xlsx"))

                if fitting_model == "NNLS":  # For NNLS perform AUC
                    d_AUC, f_AUC = data.fit_params.apply_AUC_to_results(
                        data.fit_results
                    )
                    data.fit_results.save_results_to_excel(
                        Path(out_path + "_pixel_AUC.xlsx"), d_AUC, f_AUC
                    )

                # Fit segmentation-wise
                data.fit_segmentation_wise(multi_threading=False)
                data.fit_results.save_results_to_excel(
                    Path(out_path + "_segmentation.xlsx")
                )

                if fitting_model == "NNLS":  # For NNLS perform AUC
                    d_AUC, f_AUC = data.fit_params.apply_AUC_to_results(
                        data.fit_results
                    )
                    data.fit_results.save_results_to_excel(
                        Path(out_path + "_segmentation_AUC.xlsx"), d_AUC, f_AUC
                    )

    print("Done")
