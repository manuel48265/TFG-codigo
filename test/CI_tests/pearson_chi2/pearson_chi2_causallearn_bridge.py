# Calls the installed causal-learn dependency at runtime — no source from
# that package is copied or redistributed here.

import pandas as pd
from causallearn.utils.cit import CIT

from src.CI_tests.common import CIResult


def pearson_chi2_test_causallearn(
    x: pd.Series,
    y: pd.Series,
    z: pd.DataFrame | None = None,
    alpha: float = 0.05,
) -> CIResult:
    """Call causal-learn's `chisq` CI test.

    🇬🇧 A bridge, not a reimplementation: hands the same inputs to
    causal-learn's actual `Chisq_or_Gsq` and wraps its output in the shared
    `CIResult` contract. Its public `__call__` only returns a p-value (no
    raw statistic), so `statistic` is left as None rather than recomputed
    here from its internals.

    🇪🇸 Un puente, no una reimplementación: le pasa las mismas entradas al
    `Chisq_or_Gsq` real de causal-learn y envuelve su salida en el contrato
    compartido `CIResult`. Su `__call__` público solo devuelve un p-valor
    (sin estadístico crudo), así que `statistic` se deja en None en vez de
    recalcularlo aquí a partir de sus internos.

    Args:
        x: Test variable (categorical) — Variable a testear (categórica).
        y: Test variable (categorical) — Variable a testear (categórica).
        z: Conditioning set, or None — Conjunto de condicionamiento, o None.
        alpha: Significance level — Nivel de significación.

    Returns:
        CIResult with causal-learn's p-value and reject decision —
        CIResult con el p-valor y la decisión de rechazo de causal-learn.
    """
    columns = [x.rename("x"), y.rename("y")] + ([z] if z is not None else [])
    data = pd.concat(columns, axis=1).to_numpy(dtype=object)

    dim_z = 0 if z is None else z.shape[1]
    cit = CIT(data, "chisq")
    p_value = float(cit(0, 1, tuple(range(2, 2 + dim_z))))
    return CIResult(statistic=None, p_value=p_value, reject=(p_value < alpha))
