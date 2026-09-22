import numpy as np
from numpy.typing import NDArray


def center_kernel_matrix(X: NDArray) -> NDArray[np.float64]:
    n = np.shape(X)[0]
    X_colsums = X.sum(axis=0)
    X_allsum = X_colsums.sum()
    return X - (X_colsums[None, :] + X_colsums[:, None]) / n + (X_allsum / n**2)
