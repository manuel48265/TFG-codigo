import numpy as np
import pytest

from test.df_scenarios import DF_SCENARIOS, DataFrameScenario
from test.methods import METHODS, RegisteredMethod

EPSILON = 1e-6
LOG_EPSILON = 1e-3
RNG_SEED = 0

CASES = [(method, scenario) for method in METHODS for scenario in DF_SCENARIOS]


@pytest.mark.parametrize("method,scenario", CASES, ids=[f"{m.name}-{s.name}" for m, s in CASES])
def test_python_matches_reference(method: RegisteredMethod, scenario: DataFrameScenario) -> None:
    """Every method must match its reference on every scenario it can handle.

    The DataFrame of a scenario is generated once and preprocessed by the
    method's ``prepare`` step, so the implementation and its reference always
    see identical inputs. ``statistic`` is compared only when both sides
    provide one. The global RNG is reset to the same seed before each call,
    since some methods (KCI with ``approx=False``) sample from it directly.
    """
    data = method.prepare(scenario.generate(np.random.default_rng(RNG_SEED)))

    np.random.seed(RNG_SEED)
    py = method.python_impl(data, "x", "y", scenario.conditioning)
    np.random.seed(RNG_SEED)
    ref = method.reference_impl(data, "x", "y", scenario.conditioning)

    if py.statistic is not None and ref.statistic is not None:
        assert py.statistic == pytest.approx(ref.statistic, abs=EPSILON)
    if method.compare_log_p:
        assert np.log(py.p_value) == pytest.approx(np.log(ref.p_value), abs=LOG_EPSILON)
    else:
        assert py.p_value == pytest.approx(ref.p_value, abs=EPSILON)
    assert py.reject == ref.reject
