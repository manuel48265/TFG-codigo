import pandas as pd

from src.Experiments.DAG_Generation.fundamental_topology_generator import FundamentalCase
from src.Experiments.Experimental_cases.run_experiment import Case, _generate


def _case(n: int) -> Case:
    return Case(FundamentalCase.COLLIDER, 2, n, "additive_noise", "half_categorical", 3)


def test_sample_sizes_share_seed_and_are_nested() -> None:
    small, large = _case(500), _case(1000)
    assert small.dataset_id != large.dataset_id
    assert small.seed == large.seed

    generated_small, generated_large = _generate(small), _generate(large)
    assert generated_small is not None and generated_large is not None
    data_small, types_small, attempts_small = generated_small
    data_large, types_large, attempts_large = generated_large
    assert (types_small, attempts_small) == (types_large, attempts_large)
    assert data_small.roles == data_large.roles
    pd.testing.assert_frame_equal(
        data_small.data.iloc[:500].reset_index(drop=True),
        data_large.data.iloc[:500].reset_index(drop=True),
    )
    assert len(data_large.data) == 1000
