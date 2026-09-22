import numpy as np
import pytest

from test.methods import METHODS, RegisteredMethod
from test.scenarios import SCENARIOS, Scenario

EPSILON = 1e-6
RNG_SEED = 0


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
@pytest.mark.parametrize("method", METHODS, ids=lambda m: m.name)
def test_python_matches_reference(method: RegisteredMethod, scenario: Scenario) -> None:
    """Every registered method must match its reference on every scenario.

    🇬🇧 Every scenario in `SCENARIOS` is generated once and adapted by each
    method's own `build_kwargs` — the Python port and the reference
    implementation always see identical inputs, so an exact match up to
    floating-point tolerance is expected, not just "same reject decision".
    `statistic` is compared only when both sides provide one — a reference
    whose public API doesn't expose it (e.g. causal-learn's FisherZ) sets
    it to None instead of having its bridge reconstruct it. The global RNG
    is reset to the same seed right before each call, since some methods
    (KCI with `approx=False`) sample from it directly instead of taking
    their own generator.

    🇪🇸 Cada escenario de `SCENARIOS` se genera una vez y lo adapta el
    `build_kwargs` propio de cada método — el port en Python y la
    implementación de referencia ven siempre entradas idénticas, así que se
    espera coincidencia exacta salvo tolerancia de punto flotante, no solo
    "la misma decisión de rechazo". `statistic` solo se compara cuando
    ambos lados lo dan — una referencia cuya API pública no lo expone (p.
    ej. FisherZ de causal-learn) lo deja en None en vez de que su bridge lo
    reconstruya. El RNG global se reinicia a la misma semilla justo antes
    de cada llamada, porque algunos métodos (KCI con `approx=False`)
    muestrean de él directamente en vez de recibir un generador propio.
    """
    x, y, z = scenario.generate(np.random.default_rng(RNG_SEED))
    kwargs = method.build_kwargs(x, y, z)

    np.random.seed(RNG_SEED)
    py = method.python_impl(**kwargs)
    np.random.seed(RNG_SEED)
    ref = method.reference_impl(**kwargs)

    if py.statistic is not None and ref.statistic is not None:
        assert py.statistic == pytest.approx(ref.statistic, abs=EPSILON)
    assert py.p_value == pytest.approx(ref.p_value, abs=EPSILON)
    assert py.reject == ref.reject
