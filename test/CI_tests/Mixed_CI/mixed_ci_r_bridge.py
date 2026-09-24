# Calls the installed MXM dependency at runtime — no source from that
# package is copied or redistributed here.

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import default_converter, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from src.CI_tests.common import CIResult

# Same empty-site-library cleanup as the GCM bridge — silences R's harmless
# "libraries ... contain no packages" warning on every import.
ro.r(".libPaths(Filter(function(p) length(list.files(p)) > 0, .libPaths()), include.site = FALSE)")

_conversion = default_converter + pandas2ri.converter
_mxm = importr("MXM")


def mixed_ci_test_r(
    data: pd.DataFrame,
    x: str,
    y: str,
    z: list[str] | None = None,
    alpha: float = 0.05,
) -> CIResult:
    """Call the official R package `MXM::ci.mm`.

    🇬🇧 A bridge, not a reimplementation: hands the same data to the real R
    source (via rpy2) and wraps its output in the shared `CIResult`
    contract. Columns are renamed to collision-free tags before crossing
    into R: `ci.mm`'s own code does `y <- dat[, ind1]` then later
    `lm(y ~ ., data = ds1)` — if a column in `dat` is literally named "y"
    (or "x", for the `ind2`/local-`x` side), R's formula resolves `y` from
    `ds1` instead of the intended local variable, silently fitting the
    wrong model. Verified against the installed MXM 1.5.8 by comparing
    identical data under both naming schemes.

    🇪🇸 Un puente, no una reimplementación: le pasa los mismos datos al
    código fuente R real (vía rpy2) y envuelve su salida en el contrato
    compartido `CIResult`. Las columnas se renombran a etiquetas sin
    colisión antes de cruzar a R: el propio código de `ci.mm` hace
    `y <- dat[, ind1]` y luego `lm(y ~ ., data = ds1)` — si una columna de
    `dat` se llama literalmente "y" (o "x", para el lado `ind2`/`x` local),
    la fórmula de R resuelve `y` desde `ds1` en vez de la variable local
    prevista, ajustando en silencio el modelo equivocado. Verificado contra
    MXM 1.5.8 instalado comparando los mismos datos bajo ambos esquemas de
    nombres.

    Args:
        data: Full dataset — Conjunto de datos completo.
        x: Column name of the first test variable — Nombre de columna de la
            primera variable a testear.
        y: Column name of the second test variable — Nombre de columna de
            la segunda variable a testear.
        z: Column names of the conditioning set, or None — Nombres de
            columna del conjunto de condicionamiento, o None.
        alpha: Significance level — Nivel de significación.

    Returns:
        CIResult with `ci.mm`'s p-value and reject decision; `statistic` is
        None (R's `2*max(t1,t2)`-style stat has no theoretical backing in
        the paper, see `mixed_ci.py`) — CIResult con el p-valor y la
        decisión de rechazo de `ci.mm`; `statistic` es None (el estadístico
        de R, tipo `2*max(t1,t2)`, no tiene respaldo teórico en el paper,
        ver `mixed_ci.py`).
    """
    cs_rename = {col: f"r_cs{i}" for i, col in enumerate(z)} if z else {}
    rename = {x: "r_ind1", y: "r_ind2"} | cs_rename
    r_ready = data.rename(columns=rename)

    with localconverter(_conversion):
        r_df = ro.conversion.get_conversion().py2rpy(r_ready)
        cs = ro.NULL if z is None else ro.StrVector([rename[c] for c in z])
        result = _mxm.ci_mm("r_ind1", "r_ind2", cs, r_df)
        _, log_p, _ = list(result)

    p_value = float(np.exp(log_p))
    return CIResult(statistic=None, p_value=p_value, reject=(p_value < alpha))
