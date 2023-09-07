import numpy as np
from scipy.optimize import curve_fit, nnls
from src.fit.NNLSregCV import NNLSregCV

# from fit import FitData


class Model(object):
    """Model class returning fit of selected model with applied parameters"""

    class BasicModel(object):
        def __init__(self, max_iter: int | None = 200):
            self.max_iter = max_iter

        def fit(self, *args):
            pass

    class NNLS(BasicModel):
        def __init__(self, max_iter: int = 200):
            super().__init__(max_iter=max_iter)

        def fit(self, idx: int, signal: np.ndarray, basis: np.ndarray):
            """NNLS fitting model (may include regularisation)"""

            fit, _ = nnls(basis, signal, maxiter=self.max_iter)
            return idx, fit

    class NNLSRegCV(BasicModel):
        def __init__(self, tol: float = 0.0001):
            super().__init__(max_iter=None)
            self.tol = tol

        def fit(self, idx: int, signal: np.ndarray, basis: np.ndarray):
            """NNLS fitting model with cross-validation algorithm for automatic regularisation weighting"""

            fit, _, _ = NNLSregCV(basis, signal, self.tol)
            return idx, fit

    class MultiExp(BasicModel):
        """Multi-exponential fitting model (for non-linear fitting methods and algorithms)"""

        def __init__(
            self,
            n_components: int | None = None,
            max_iter: int | None = None,
            TM: float | None = None,
        ):
            super().__init__(max_iter=max_iter)
            self._n_components = n_components
            self.TM = TM
            if n_components:
                self.model = self.multi_exp_wrapper()
            else:
                self.model = None

        @property
        def n_components(self):
            return self._n_components

        @n_components.setter
        def n_components(self, value: int):
            self._n_components = value
            self.model = self.multi_exp_wrapper()

        @property
        def TM(self):
            return self._TM

        @TM.setter
        def TM(self, value: float):
            self._TM = value
            self.model = self.multi_exp_wrapper()

        @staticmethod
        def print_model(n_components, args):
            return Model.multi_exp_printer(n_components, args)

        def multi_exp_wrapper(self):
            def multi_exp_model(b_values, *args):
                f = 0
                for i in range(self.n_components - 1):
                    f += (
                        np.exp(-np.kron(b_values, abs(args[i])))
                        * args[self.n_components + i]
                    )
                f += (
                    np.exp(-np.kron(b_values, abs(args[self.n_components - 1])))
                    # Second half containing f, except for S0 as the very last entry
                    * (1 - (np.sum(args[self.n_components : -1])))
                )

                if self.TM:
                    # With nth entry being T1 in cases of T1 fitting
                    f *= np.exp(-args[self.n_components] / self.TM)

                return f * args[-1]  # Add S0 term for non-normalized signal

            return multi_exp_model

        def fit(
            self,
            idx: int,
            signal: np.ndarray,
            b_values: np.ndarray,
            args: np.ndarray,
            lb: np.ndarray,
            ub: np.ndarray,
        ):
            fit = curve_fit(
                self.model,
                b_values,
                signal,
                args,
                bounds=(lb, ub),
                max_nfev=self.max_iter,
            )[0]
            return idx, fit

    @staticmethod
    def multi_exp_printer(n_components: int, args):
        f = f""
        for i in range(n_components - 1):
            f += f"exp(-kron(b_values, abs({args[i]}))) * {args[n_components + i]} + "
        f += f"exp(-kron(b_values, abs({args[n_components-1]}))) * (1 - (sum({args[n_components:-1]})))"
        return f"( " + f + f" ) * {args[-1]}"
