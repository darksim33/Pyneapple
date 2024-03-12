import os.path
from pathlib import Path
from multiprocessing import freeze_support
from src.utils import Nii, NiiSeg
from src.fit.fit import FitData
import glob
from tqdm import tqdm

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
    folder_path = "data/MEDIA_data"
    fitting_models = ["IVIM", "NNLS", "IDEAL"]
    fitting_parameters = None

    # Filter path for img (.nii) and seg files (.nii.gz)
    for img, seg in (
        glob.glob(folder_path + "/*.nii"),
        glob.glob(folder_path + "/*.nii.gz"),
    ):
        img_files = img
        seg_files = seg

    # Fitting all models to every segmentation of each image
    for img_file in tqdm(img_files, desc=" image", position=0, total=len(img_files)):
        # Assign subject and scan specific segmentations based on image filename
        seg_files_subject = [seg for seg in seg_files if Path(img_file).stem in seg]

        for seg_file in tqdm(
            seg_files_subject,
            desc=" segmentation",
            position=1,
            leave=False,
            total=len(seg_files_subject),
        ):
            for model, fitting_model in enumerate(fitting_models):
                # Initiate fitting procedure
                data = FitData(
                    model=fitting_model, img=Nii(img_file), seg=NiiSeg(seg_file)
                )

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

                if fitting_model == "NNLS":
                    d_AUC, f_AUC = data.fit_params.apply_AUC_to_results(
                        data.fit_results
                    )
                    data.fit_results.save_results_to_excel(
                        Path(out_path + "_segmentation_AUC.xlsx"), d_AUC, f_AUC
                    )

    print("Done")
