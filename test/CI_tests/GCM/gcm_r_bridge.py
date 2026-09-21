import numpy as np
import rpy2.robjects.numpy2ri as numpy2ri
from numpy.typing import NDArray
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from src.CI_tests.common import CIResult

_conversion = default_converter + numpy2ri.converter
_gcm_r = importr("GeneralisedCovarianceMeasure")


def gcm_test_r(
    x: NDArray[np.float64] | None = None,
    y: NDArray[np.float64] | None = None,
    z: NDArray[np.float64] | None = None,
    alpha: float = 0.05,
    nsim: int = 499,
    resid_x_on_z: NDArray[np.float64] | None = None,
    resid_y_on_z: NDArray[np.float64] | None = None,
) -> CIResult:
    """Call the official R package `GeneralisedCovarianceMeasure::gcm.test`.

    🇬🇧 A bridge, not a reimplementation: hands the same inputs to the
    real R source (via rpy2) and wraps its output in the shared
    `CIResult` contract, so it can be compared against `gcm_test` (the
    Python port) using identical call sites in tests.

    🇪🇸 Un puente, no una reimplementación: le pasa las mismas entradas
    al código fuente R real (vía rpy2) y envuelve su salida en el
    contrato compartido `CIResult`, para poder compararlo con `gcm_test`
    (el port a Python) usando exactamente las mismas llamadas en los tests.

    Args:
        x: Test variable, shape (n,) — Variable a testear, forma (n,).
        y: Test variable, shape (n,) — Variable a testear, forma (n,).
        z: Conditioning set, shape (n, dz) — Conjunto de condicionamiento,
            forma (n, dz).
        alpha: Significance level — Nivel de significación.
        nsim: Number of multiplier-bootstrap draws (multivariate case
            only) — Número de repeticiones del bootstrap multiplicador
            (solo caso multivariante).
        resid_x_on_z: Precomputed residuals of x on z — Residuos
            precalculados de x sobre z.
        resid_y_on_z: Precomputed residuals of y on z — Residuos
            precalculados de y sobre z.

    Returns:
        CIResult with the R package's statistic, p-value and reject
        decision — CIResult con el estadístico, p-valor y decisión de
        rechazo del paquete R.
    """
    r_args: dict[str, object] = {"alpha": alpha, "nsim": nsim}
    if resid_x_on_z is not None and resid_y_on_z is not None:
        r_args["resid.XonZ"] = resid_x_on_z
        r_args["resid.YonZ"] = resid_y_on_z
    else:
        r_args["X"] = x
        r_args["Y"] = y
        r_args["Z"] = z

    with localconverter(_conversion):
        result = _gcm_r.gcm_test(**r_args)
        return CIResult(
            statistic=float(result.getbyname("test.statistic")[0]),
            p_value=float(result.getbyname("p.value")[0]),
            reject=bool(result.getbyname("reject")[0]),
        )
