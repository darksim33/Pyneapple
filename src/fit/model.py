import numpy as np
from scipy.optimize import least_squares, curve_fit, nnls
from .NNLSregCV import NNLSregCV
# from fit import FitData


class Model(object):
    """Model class returning fit of selected model with applied parameters"""

    def NNLS(idx: int, signal: np.ndarray, basis: np.ndarray, max_iter: int = 200):
        """NNLS fitting model (may include regularisation)"""

        fit, _ = nnls(basis, signal, maxiter=max_iter)
        return idx, fit

    def NNLS_reg_CV(
        idx: int, signal: np.ndarray, basis: np.ndarray, tol: float = 0.0001
    ):
        """NNLS fitting model with cross-validation algorithm for automatic regularisation weighting"""

        fit, _, _ = NNLSregCV(basis, signal, tol)
        return idx, fit

    def mono(
        idx: int,
        signal: np.ndarray,
        b_values: np.ndarray,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        max_iter: int,
        TM: float | None,
    ):
        """Mono exponential fitting model for ADC and T1"""
        # NOTE does not theme to work for T1

        def mono_wrapper(TM: float | None):
            # TODO: use multi_exp(n=1)
            def mono_model(
                b_values: np.ndarray,
                S0: float | int,
                x0: float | int,
                T1: float | int = 0,
            ):
                if TM is None or 0:
                    return np.array(S0 * np.exp(-np.kron(b_values, x0)))

                return np.array(S0 * np.exp(-np.kron(b_values, x0)) * np.exp(-T1 / TM))

            return mono_model

        fit, _ = curve_fit(
            mono_wrapper(TM),
            b_values,
            signal,
            x0,
            bounds=(lb, ub),
            max_nfev=max_iter,
        )
        return idx, fit

    def multi_exp(
        idx: int,
        signal: np.ndarray,
        b_values: np.ndarray,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        n_components: int,
        max_iter: int,
    ):
        """Multiexponential fitting model (e.g. for NLLS, mono, IDEAL ...)"""

        def multi_exp_wrapper(n_components: int):
            def multi_exp_model(b_values: np.ndarray, x0: float | int):
                f = 0
                for i in range(n_components - 2):
                    f = +np.exp(-np.kron(b_values, abs(x0[i]))) * x0[n_components + i]
                return f + np.exp(-np.kron(b_values, abs(x0[n_components - 1]))) * (
                    100 - (np.sum(x0[n_components:]))
                )

            return multi_exp_model

        fit, _ = curve_fit(
            multi_exp_wrapper(n_components=n_components),
            b_values,
            signal,
            x0,
            bounds=(lb, ub),
        )
        return idx, fit

# TODO: Solve inheritance Porblem -> move Parameters?!
# class NNLSParams(FitData.Parameters):
#     def __init__(
#         self,
#         # TODO: model & b_values should be inherited without additional initialisation
#         model: Model | None = Model.NNLS,
#         max_iter: int
#         | None = 250,  # TODO: necessary? only for re-initialising model...
#         n_bins: int | None = 250,
#         d_range: np.ndarray | None = np.array([1 * 1e-4, 2 * 1e-1]),
#         # n_pools: int | None = 4,
#     ):
#         """
#         Basic NNLS Parameter Class
#         model: should be of class Model
#         """

#         if not model:
#             super().__init__(Model.NNLS)
#         else:
#             super().__init__(model, max_iter=max_iter)
#         self.boundaries.n_bins = n_bins
#         self.boundaries.d_range = d_range

#     def get_basis(self) -> np.ndarray:
#         self._basis = np.exp(
#             -np.kron(
#                 self.b_values.T,
#                 self.get_bins(),
#             )
#         )
#         return self._basis

#     def get_fit_function(self):
#         return partial(self.model, basis=self.get_basis())

#     def eval_pixelwise_fitting_results(self, results_pixel, seg) -> FitData.Results:
#         # Create output array for spectrum
#         new_shape = np.array(seg.array.shape)
#         new_shape[3] = self.get_basis().shape[1]
#         fit_results = FitData.Results()
#         fit_results.spectrum = np.zeros(new_shape)
#         # Sort entries to array
#         for pixel in results_pixel:
#             fit_results.spectrum[pixel[0]] = pixel[1]
#         # TODO: add d and f and implement find_peaks
#         return fit_results


# class NNLSregParams(NNLSParams):
#     def __init__(
#         self,
#         model,
#         reg_order: int
#         | None = 2,  # TODO: fuse NNLSParams (reg=0) and NNLSregParams (reg!=0)?
#         mu: float | None = 0.01,
#     ):
#         super().__init__(
#             model=model,
#             max_iter=100000,
#         )
#         self.reg_order = reg_order
#         self.mu = mu

#     def get_basis(self) -> np.ndarray:
#         basis = super().get_basis()
#         n_bins = self.boundaries.n_bins

#         if self.reg_order == 0:
#             # no weighting
#             reg = diags([1], [0], shape=(n_bins, n_bins)).toarray()
#         elif self.reg_order == 1:
#             # weighting with the predecessor
#             reg = diags([-1, 1], [0, 1], shape=(n_bins, n_bins)).toarray()
#         elif self.reg_order == 2:
#             # weighting of the nearest neighbours
#             reg = diags([1, -2, 1], [-1, 0, 1], shape=(n_bins, n_bins)).toarray()
#         elif self.reg_order == 3:
#             # weighting of the first and second nearest neighbours
#             reg = diags(
#                 [1, 2, -6, 2, 1], [-2, -1, 0, 1, 2], shape=(n_bins, n_bins)
#             ).toarray()

#         # append reg to create regularised NNLS basis
#         return np.concatenate((basis, reg * self.mu))

#     def get_pixel_args(self, img: Nii, seg: Nii_seg, debug: bool):
#         # enhance image array for regularisation
#         reg = np.zeros((np.append(np.array(img.array.shape[0:3]), 250)))
#         img_reg = np.concatenate((img.array, reg), axis=3)

#         # TODO: understand code @TT
#         if debug:
#             pixel_args = zip(
#                 (
#                     ((i, j, k), img_reg[i, j, k, :])
#                     for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
#                 )
#             )
#         else:
#             pixel_args = zip(
#                 (
#                     (i, j, k)
#                     for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
#                 ),
#                 (
#                     img_reg[i, j, k, :]
#                     for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
#                 ),
#             )

#         return pixel_args


# class NNLSregCVParams(NNLSParams):
#     def __init__(
#         self, model: Model | None = Model.NNLS_reg_CV, tol: float | None = 0.0001
#     ):
#         super().__init__(model=model)
#         self.tol = tol


# class MonoParams(FitData.Parameters):
#     def __init__(
#         self,
#         model: Model | None = Model.mono,
#         x0: np.ndarray | None = np.array([50, 0.001]),
#         lb: np.ndarray | None = np.array([10, 0.0001]),
#         ub: np.ndarray | None = np.array([1000, 0.01]),
#         TM: int | None = None,
#         max_iter: int
#         | None = 600,  # TODO: None überflüssig? schon in Parameters gesetzt
#     ):
#         super().__init__(model=model, max_iter=max_iter)
#         self.boundaries.x0 = x0
#         self.boundaries.lb = lb
#         self.boundaries.ub = ub
#         self.variables.TM = TM

#     # why function and not just self.b_values? @TT
#     def get_basis(self):
#         # BUG Bvlaues are passed in the wrong shape
#         return np.squeeze(self.b_values)

#     def get_fit_function(self):
#         return partial(
#             self.model,
#             b_values=self.get_basis(),
#             x0=self.boundaries.x0,
#             lb=self.boundaries.lb,
#             ub=self.boundaries.ub,
#             TM=self.variables.TM,
#             max_iter=self.max_iter,
#         )

#     def eval_pixelwise_fitting_results(self, results_pixel, seg) -> FitData.Results:
#         # prepare arrays
#         fit_results = FitData.Results()
#         for pixel in results_pixel:
#             fit_results.S0.append((pixel[0], [pixel[1][0]]))
#             fit_results.d.append((pixel[0], [pixel[1][1]]))
#             fit_results.f.append((pixel[0], np.ones(1)))

#         # NOTE for T1 just all super and then load results again for aditional T1 values

#         fit_results = self.set_spectrum_from_variables(fit_results, seg)

#         return fit_results

#     def set_spectrum_from_variables(self, fit_results: FitData.Results, seg: Nii_seg):
#         # adjust D values acording to bins/dvalues
#         d_values = self.get_bins()
#         d_new = np.zeros(len(fit_results.d[1]))

#         new_shape = np.array(seg.array.shape)
#         new_shape[3] = self.boundaries.n_bins
#         spectrum = np.zeros(new_shape)

#         for d_pixel, f_pixel in zip(fit_results.d, fit_results.f):
#             temp_spec = np.zeros(self.boundaries.n_bins)
#             for idx, (D, F) in enumerate(zip(d_pixel[1], f_pixel[1])):
#                 index = np.unravel_index(
#                     np.argmin(abs(d_values - D), axis=None),
#                     d_values.shape,
#                 )[0].astype(int)
#                 d_new[idx] = d_values[index]
#                 temp_spec = temp_spec + F * signal.unit_impulse(
#                     self.boundaries.n_bins, index
#                 )
#             spectrum[d_pixel[0]] = temp_spec
#         fit_results.spectrum = spectrum
#         return fit_results


# class MonoT1Params(MonoParams):
#     def __init__(
#         self,
#         model: Model | None = Model.mono,
#         x0: np.ndarray | None = None,
#         lb: np.ndarray | None = None,
#         ub: np.ndarray | None = None,
#         TM: float | None = None,
#         max_iter: int | None = 600,
#     ):
#         super().__init__(model=model, max_iter=max_iter)
#         # Welches model denn sonst? Abfrage geschieht doch schon vorher?! -> move up @TT
#         if model == Model.mono:
#             self.boundaries.x0 = x0 if x0 is not None else np.array([50, 0.001, 1750])
#             self.boundaries.lb = lb if lb is not None else np.array([10, 0.0001, 1000])
#             self.boundaries.ub = ub if ub is not None else np.array([1000, 0.01, 2500])
#             self.variables.TM = TM if TM is not None else 20.0

#     # TODO: same as in MonoT1 and NNLS -> inherit functions?
#     def get_fit_function(self):
#         return partial(
#             self.model,
#             b_values=self.get_basis(),
#             x0=self.boundaries.x0,
#             lb=self.boundaries.lb,
#             ub=self.boundaries.ub,
#             TM=self.variables.TM,
#             max_iter=self.max_iter,
#         )

#     def eval_pixelwise_fitting_results(self, results_pixel, seg) -> FitData.Results:
#         fit_results = super().eval_pixelwise_fitting_results(results_pixel, seg)
#         for pixel in results_pixel:
#             fit_results.T1.append((pixel[0], [pixel[1][2]]))
#         return fit_results


# # def setup_signalbased_fitting(fit_data: FitData):
# #     img = fit_data.img
# #     seg = fit_data.seg
# #     fit_results = list()
# #     for seg_idx in range(1, seg.number_segs + 1, 1):
# #         img_seg = seg.get_single_seg_mask(seg_idx)
# #         signal = Processing.get_mean_seg_signal(img, img_seg, seg_idx)
# #         fit_results.append(fit_segmentation_signal(signal, fit_data, seg_idx))
# #     if fit_data.fit_params.model == Model.NNLS:
# #         # Create output array for spectrum
# #         new_shape = np.array(seg.array.shape)
# #         basis = np.exp(
# #             -np.kron(
# #                 fit_data.fit_params.b_values.T,
# #                 fit_data.fit_params.get_bins(),
# #             )
# #         )
# #         new_shape[3] = basis.shape[1]
# #         img_results = np.zeros(new_shape)
# #         # Sort Entries to array
# #         for seg in fit_results:
# #             img_results[seg[0]] = seg[1]


# # def fit_segmentation_signal(
# #     signal: np.ndarray, fit_params: FitData.Parameters, seg_idx: int
# # ):
# #     if fit_params.model == Model.NNLS:
# #         basis = np.exp(
# #             -np.kron(
# #                 fit_params.b_values.T,
# #                 fit_params.get_bins(),
# #             )
# #         )
# #         fit_function = partial(fit_params.model, basis=basis)
# #     elif fit_params.model == Model.mono:
# #         print("test")
# #     return fit_function(seg_idx, signal)