import numpy as np
from numpy.typing import NDArray


class OLSRegressor:
    """Ordinary least squares with intercept, solved in closed form.

    Deterministic, so two implementations residualizing with it see identical residuals.
    """

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "OLSRegressor":
        design = np.column_stack([np.ones(X.shape[0]), X])
        self.coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.column_stack([np.ones(X.shape[0]), X]) @ self.coefficients
