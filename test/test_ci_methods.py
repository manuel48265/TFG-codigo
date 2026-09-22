import numpy as np
import pytest

from test.methods import METHODS, RegisteredMethod
from test.scenarios import SCENARIOS, ResidualScenario

EPSILON = 1e-6


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
@pytest.mark.parametrize("method", METHODS, ids=lambda m: m.name)
def test_python_matches_reference(method: RegisteredMethod, scenario: ResidualScenario) -> None:
    """Every registered method must match its reference on every scenario.

    🇬🇧 Feeds the same precomputed residuals to the Python port and the
    reference implementation, so the comparison is fully deterministic
    (no regressor/RNG involved) — an exact match up to floating-point
    tolerance is expected, not just "same reject decision".

    🇪🇸 Alimenta los mismos residuos precalculados al port en Python y a
    la implementación de referencia, para que la comparación sea
    totalmente determinista (sin regresor ni aleatoriedad de por medio)
    — se espera coincidencia exacta salvo tolerancia de punto flotante,
    no solo "la misma decisión de rechazo".
    """
    resid_x, resid_y = scenario.generate(np.random.default_rng(0))

    py = method.python_impl(resid_x_on_z=resid_x, resid_y_on_z=resid_y)
    ref = method.reference_impl(resid_x_on_z=resid_x, resid_y_on_z=resid_y)

    assert py.statistic == pytest.approx(ref.statistic, abs=EPSILON)
    assert py.p_value == pytest.approx(ref.p_value, abs=EPSILON)
    assert py.reject == ref.reject
