from pathlib import Path
import time

from ..utils.nifti import Nii, NiiSeg
from . import parameters
from ..utils.multithreading import multithreader
from .IDEAL import fit_IDEAL
from .results import Results


class FitData:
    """
    Fitting class for (multi-threaded) pixel- and segmentation-wise fitting.

    Attributes
    ----------
    model : str
    img : Nii
    seg : NiiSeg

    Methods
    --------
    fit_pixel_wise(multi_threading: bool | None = True)
        Fits every pixel inside the segmentation individually. Multi-threading possible to boost performance.
    fit_segmentation_wise()
        Fits mean signal of segmentation(s).
    """

    def __init__(
        self,
        model: str | None = None,
        params_json: str | Path | None = None,
        img: Nii | None = Nii(),
        seg: NiiSeg | None = NiiSeg(),
    ):
        self.model_name = model
        self.params_json = params_json
        self.img = img
        self.seg = seg
        if model == "NNLS":
            self.params = parameters.NNLSParams(params_json)
        elif model == "NNLSCV":
            self.params = parameters.NNLSCVParams(params_json)
        elif model == "IVIM":
            self.params = parameters.IVIMParams(params_json)
        elif model == "IDEAL":
            self.params = parameters.IDEALParams(params_json)
        else:
            self.params = parameters.Parameters(params_json)
            # print("Warning: No valid Fitting Method selected")
        self.results = Results()
        self.flags = dict()
        self.set_default_flags()

    def set_default_flags(self):
        self.flags["did_fit"] = False

    def reset(self):
        self.model_name = None
        self.img = Nii()
        self.seg = NiiSeg()
        self.params = parameters.Parameters()
        self.results = Results()
        self.set_default_flags()

    def fit_pixel_wise(self, multi_threading: bool | None = True):
        """Fits every pixel inside the segmentation individually."""
        if self.params.json is not None and self.params.b_values is not None:
            print(f"Fitting {self.model_name} pixel wise...")
            start_time = time.time()
            pixel_args = self.params.get_pixel_args(self.img, self.seg)

            results = multithreader(
                self.params.fit_function,
                pixel_args,
                self.params.n_pools if multi_threading else None,
            )

            results_dict = self.params.eval_fitting_results(results)
            self.results.update_results(results_dict)
            print(f"Pixel-wise fitting time: {round(time.time() - start_time, 2)}s")
        else:
            ValueError("No valid Parameter Set for fitting selected!")

    def fit_segmentation_wise(self):
        """Fits mean signal of segmentation(s), computed of all pixels signals."""
        if self.params.json is not None and self.params.b_values is not None:
            print(f"Fitting {self.model_name} segmentation wise...")
            start_time = time.time()
            results = list()
            for seg_number in self.seg.seg_numbers.astype(int):
                # get mean pixel signal
                seg_args = self.params.get_seg_args(self.img, self.seg, seg_number)
                # fit mean signal
                seg_results = multithreader(
                    self.params.fit_function,
                    seg_args,
                    n_pools=None,  # self.params.n_pools,
                )

                # Save result of mean signal for every pixel of each seg
                # for pixel in self.seg.get_seg_indices(seg_number):
                #     results.append((pixel, seg_results[0][1]))
                results.append((seg_number, seg_results[0][1]))

            self.results.set_segmentation_wise(self.seg.seg_indices)

            results_dict = self.params.eval_fitting_results(results)
            self.results.update_results(results_dict)

            print(
                f"Segmentation-wise fitting time: {round(time.time() - start_time, 2)}s"
            )
        else:
            ValueError("No valid Parameter Set for fitting selected!")

    def fit_IDEAL(self, multi_threading: bool = False, debug: bool = False):
        """IDEAL Fitting Interface."""
        start_time = time.time()
        if not self.model_name == "IDEAL":
            raise AttributeError("Wrong model name!")
        print(f"The initial image size is {self.img.array.shape[0:4]}.")
        fit_results = fit_IDEAL(self.img, self.seg, self.params, multi_threading, debug)
        self.results = self.params.eval_fitting_results(fit_results)
        print(f"IDEAL fitting time:{round(time.time() - start_time, 2)}s")

    def fit_ivim_segmented(self, multi_threading: bool = False, debug: bool = False):
        self.params: parameters.IVIMSegmentedParams
        start_timer = time.time()
        if not self.model_name == "IVIM_segmented":
            raise AttributeError("Wrong model name!")
        print("Fitting first component for segmented IVIM model...")

        # Get Pixel Args for first Fit
        pixel_args = self.params.get_pixel_args_fixed(self.img, self.seg)
        # Run Fitting
        results = multithreader(
            self.params.params_fixed.fit_function,
            pixel_args,
            self.params.n_pools if multi_threading else None,
        )
        # Eval raw Results
        results_dict = self.params.params_fixed.eval_fitting_results(results)
        # Create Results structure
        temp_results = Results()
        temp_results.update_results(results_dict)
        # Get parameter Maps
        fixed_values = self.params.get_fixed_fit_results(
            temp_results, self.img.array.shape
        )
        # Get new pixel args with
        pixel_args = self.params.get_pixel_args(self.img, self.seg, *fixed_values)

        # create new dataset from first fit

        print("Fitting all remaining components for segmented IVIM model...")

        # fit lower order model
