from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Regressor(Protocol):
    """🇬🇧 Structural type for any fit/predict regressor (sklearn, xgboost...).

    🇪🇸 Tipo estructural para cualquier regresor con fit/predict (sklearn, xgboost...).
    """

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "Regressor": ...
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...


def residuals_of(
    v: NDArray[np.float64],
    z: NDArray[np.float64],
    estimator: Regressor,
) -> NDArray[np.float64]:
    """Regress v on z and return the residuals.

    🇬🇧 Fits `estimator` on (z, v) and returns v - E[v|z], the residual
    vector needed by the GCM test statistic.

    🇪🇸 Ajusta `estimator` sobre (z, v) y devuelve v - E[v|z], el vector
    de residuos que necesita el estadístico GCM.

    Args:
        v: Variable to residualize, shape (n,) — Variable a residualizar,
            forma (n,).
        z: Conditioning set, shape (n, dz) — Conjunto de condicionamiento,
            forma (n, dz).
        estimator: Any sklearn-style regressor (e.g. `XGBRegressor()`) —
            Cualquier regresor estilo sklearn (p. ej. `XGBRegressor()`).

    Returns:
        Residual vector, shape (n,) — Vector de residuos, forma (n,).
    """
    estimator.fit(z, v)
    return v - estimator.predict(z)


def compute_residuals(
    v: NDArray[np.float64] | None,
    z: NDArray[np.float64],
    estimator: Regressor,
    precomputed: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Get residuals of v on z, computing them only if not already given.

    🇬🇧 Mirrors gcm.R's per-variable block: if `precomputed` is given, use
    it as-is; otherwise regress `v` on `z` column by column (handles both
    the univariate and the multivariate case) — R's `is.matrix(X)` +
    `apply(X, 2, resid_func)`. Raises if neither is available (R's `stop()`).

    🇪🇸 Replica el bloque por variable de gcm.R: si se da `precomputed`, se
    usa tal cual; si no, regresa `v` sobre `z` columna a columna (cubre el
    caso univariante y el multivariante) — el `is.matrix(X)` +
    `apply(X, 2, resid_func)` de R. Lanza un error si no hay ninguno de
    los dos disponibles (el `stop()` de R).

    Args:
        v: Variable to residualize, or None if `precomputed` is given —
            Variable a residualizar, o None si se da `precomputed`.
        z: Conditioning set, shape (n, dz) — Conjunto de condicionamiento,
            forma (n, dz).
        estimator: Regressor used for every column of v — Regresor usado
            para cada columna de v.
        precomputed: Residuals already computed elsewhere, skips the
            regression if given — Residuos ya calculados, se salta la
            regresión si se proporciona.

    Returns:
        Residual array, shape (n,) or (n, dv) — Array de residuos, forma
        (n,) o (n, dv).
    """
    if precomputed is not None:
        return precomputed
    if v is None:
        raise ValueError("Either the variable or its precomputed residuals must be provided.")
    if v.ndim > 1:
        return np.column_stack([residuals_of(v[:, j], z, estimator) for j in range(v.shape[1])])
    return residuals_of(v, z, estimator)
