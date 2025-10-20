"""
Example script to run an Intravoxel Incoherent Motion (IVIM) analysis using the provided functions.
"""

from pathlib import Path
from radimgarray import RadImgArray, SegImgArray
import pyneapple


def main():
    """Run the IVIM fitting example."""

    #  Set Working Directory for file processing
    working_dir = Path(__file__).parent.parent
    # Load the DWI image and segmentation data
    img = RadImgArray(working_dir / "examples" / "images" / "test_img.nii.gz")
    seg = SegImgArray(working_dir / "examples" / "images" / "test_seg.nii.gz")

    # Load the IVIM fitting parameters from a JSON or TOML file
    json = working_dir / "examples" / "parameters" / "params_biexp.toml"

    # Initiate data object with the image, segmentation and parameters
    data = pyneapple.FitData(
        img,
        seg,
        json,
    )
    # Perform the IVIM fitting
    data.fit_pixel_wise()
    # Export data to Excel and NIfTI format
    data.results.save_to_excel(working_dir / "ivim_results.xlsx")
    # to recover original image space information use img Array
    data.results.save_to_nii(working_dir / "ivim_results", img)


if __name__ == "__main__":
    main()
