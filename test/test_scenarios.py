import numpy as np
import pytest

from test.scenarios import SCENARIOS, Scenario, has_near_tie


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_clean_seed_has_no_near_tie(scenario: Scenario) -> None:
    """Cheap regression guard (data generation only, no CI test involved):
    a scenario's stored `clean_seed` must still be free of near-ties. If a
    future change to its generator invalidates it, this fails loudly
    instead of a consumer (e.g. test_cmiknn.py) silently losing coverage.
    """
    x, y, z = scenario.generate(np.random.default_rng(scenario.clean_seed))
    assert not has_near_tie(x, y, z), (
        f"{scenario.name}'s clean_seed={scenario.clean_seed} now hits a near-tie "
        "-- recompute it (see test/scenarios.py's has_near_tie)"
    )
