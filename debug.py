import sys

from pathlib import Path
from multiprocessing import freeze_support
from PyQt6 import QtCore, QtWidgets

import src.fit.parameters as params
from src.utils import Nii, NiiSeg
from src.fit.fit import FitData


# from src.fit.model import Model

from src.ui.dialogues.fitting_dlg import FittingDlg
from src.appdata import AppData
from src.ui.dialogues import prompt_dlg


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        """The Main App Window."""
        super(MainWindow, self).__init__()
        self.data = AppData()
        self.setWindowTitle("Test")

        json = params.JsonImporter(
            Path(r"resources/fitting/default_params_NNLS.json")
        )
        json.load_json()

        dlg = FittingDlg(self, params.NNLSParams())
        # dlg = FittingDlg(self, params.IVIMParams())
        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        result = dlg.exec()
        dlg.parameters.get_parameters()
        print(result)


if __name__ == "__main__":
    freeze_support()
    # app = QtWidgets.QApplication(sys.argv)
    #
    # main_window = MainWindow()
    # main_window.show()
    # sys.exit(app.exec())

    img = Nii(Path(r"data/test_img_176_176.nii"))
    seg = NiiSeg(Path(r"data/test_mask.nii.gz"))

    # nnls_json = Path(r"resources/fitting/default_params_NNLSregCV.json")
    # data_nnls = FitData("NNLSregCV", nnls_json, img=img, seg=seg)

    # nnls_json = Path(r"resources/fitting/default_params_NNLS.json")
    # data_nnls = FitData("NNLS", nnls_json, img=img, seg=seg)
    #
    # data_nnls.fit_pixel_wise(multi_threading=False)

    ivim_json = Path(r"resources/fitting/default_params_IVIM_bi.json")
    data_ivim = FitData("IVIM", ivim_json, img, seg)
    data_ivim.fit_segmentation_wise()
    # # data_ivim.fit_pixel_wise(multi_threading=False)
    # data_ivim.fit_results.save_fitted_parameters_to_nii(
    #     r"test\debug\test_ivim.nii",
    #     data_ivim.img.array.shape,
    #     dtype=float,
    #     parameter_names=data_ivim.fit_params.parameter_names,
    # )
    print("Done")

# if __name__ == "__main__":
#     start_time = time.time()
#     freeze_support()
#     # Test Set
#     img = Nii(Path(r"data/test_img_176_176.nii"))
#     seg = NiiSeg(Path(r"data/test_mask_simple_huge.nii.gz"))
#     # json = Path(
#     #     r"resources/fitting/default_params_ideal_test.json",
#     # )
#     # Prostate Set
#     # img = Nii(Path(r"data/01_prostate_img.nii"))
#     # seg = NiiSeg(Path(r"data/01_prostate_mask.nii.gz"))
#     json_ideal = Path(
#         r"resources/fitting/params_prostate_ideal.json",
#     )
#     ivim_json = Path(r"resources/fitting/params_prostate_ivim.json")
#
#     img.zero_padding()
#     # img.save("01_prostate_img_164x.nii.gz")
#     seg.zero_padding()
#     seg.array = np.fliplr(seg.array)
#     # seg.save("01_prostate_seg_164x.nii.gz")
#
#     multi_threading = True
#     # IVIM
#     data_ivim = FitData("IVIM", ivim_json, img, seg)
#     data_ivim.img.array = data_ivim.fit_params.normalize(data_ivim.img.array)
#     data_ivim.fit_params.fit_function = Model.IVIMReduced.fit
#     data_ivim.fit_pixel_wise(multi_threading=multi_threading)
#     data_ivim.fit_results.save_fitted_parameters_to_nii(
#         "test_ivim.nii", data_ivim.img.array.shape, dtype=float
#     )
#     data_ivim.fit_results.save_spectrum_to_nii("test_ivim_spectrum.nii")
#
#     stop_time = time.time() - start_time
#     # print(f"{round(stop_time, 2)}s")
#
#     # IDEAL
#     # data = FitData("IDEAL", json_ideal, img, seg)
#     # data.fit_params.n_pools = 12
#     # data.fit_ideal(multi_threading=multi_threading, debug=False)
#     # data.fit_results.save_fitted_parameters_to_nii(
#     #     "test_ideal.nii", data.img.array.shape, dtype=float
#     # )
#     # data.fit_results.save_spectrum_to_nii("test_ideal_spectrum.nii")
#     #
#     # stop_time = time.time() - stop_time
#     # print(f"{round(stop_time, 2)}s")
#     # stop_time = time.time() - stop_time
#     # print(f"{round(stop_time, 2)}s")
#     print("Done")
