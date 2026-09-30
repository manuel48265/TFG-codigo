import numpy as np
import pytest

from src.CI_tests.common import frame_to_arrays
from test.df_scenarios import CONTINUOUS_SCENARIOS, DataFrameScenario, has_near_tie


@pytest.mark.parametrize("scenario", CONTINUOUS_SCENARIOS, ids=lambda s: s.name)
def test_clean_seed_has_no_near_tie(scenario: DataFrameScenario) -> None:
    """The stored ``clean_seed`` of a scenario must still generate data without near-ties."""
    df = scenario.generate(np.random.default_rng(scenario.clean_seed))
    x, y, z = frame_to_arrays(df, "x", "y", scenario.conditioning)
    assert z is not None

    assert not has_near_tie(x, y, z), (
        f"{scenario.name}'s clean_seed={scenario.clean_seed} now hits a near-tie "
        "-- recompute it (see has_near_tie in test/df_scenarios.py)"
    )


def test_has_near_tie_flags_only_gaps_below_the_noise_scale() -> None:
    spread = np.linspace(0.0, 1.0, 50)
    tied = spread.copy()
    tied[1] = tied[0] + 1e-9

    assert not has_near_tie(spread)
    assert has_near_tie(tied)
