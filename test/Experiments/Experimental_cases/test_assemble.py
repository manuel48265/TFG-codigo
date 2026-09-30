import numpy as np
import pytest

from src.CI_tests.Fisher_Z.fisher_z import fisher_z_test
from src.CI_tests.KCI.kci import kci_test
from src.CI_tests.Mixed_CI.mixed_ci import mixed_ci_test
from src.CI_tests.pearson_chi2.pearson_chi2 import pearson_chi2_test
from src.Experiments.DAG_Generation.fundamental_topology_generator import (
    FundamentalCase,
    fundamental_topology,
)
from src.Experiments.Experimental_cases.assemble import assemble_dataset
from src.Experiments.Experimental_cases.common import (
    discretize_continuous_columns,
    has_constant_column,
    remove_unused_categories,
    to_float_columns,
)
from src.Experiments.Node_Generation.causal_models import (
    ADDITIVE_NOISE_MODELS,
    LOCAL_SCALE_NOISE_MODELS,
)
from src.Experiments.Node_Generation.utils import VariableType


def test_assemble_dataset_builds_one_column_per_variable_with_its_type() -> None:
    topology = [[], [0], [0, 1], [2]]
    types = [
        VariableType.CONTINUOUS,
        VariableType.BINARY,
        VariableType.ORDINAL,
        VariableType.NOMINAL,
    ]

    dataset = assemble_dataset(
        np.random.default_rng(0), topology, types, ADDITIVE_NOISE_MODELS, 300
    )

    assert list(dataset.data.columns) == ["v0", "v1", "v2", "v3"]
    assert len(dataset.data) == 300
    assert dataset.data["v3"].cat.ordered is False
    assert dataset.data["v2"].cat.ordered is True
    assert set(dataset.data["v1"].unique()) == {0, 1}
    assert set(dataset.roles) == {"v1", "v2", "v3"}


def test_assemble_dataset_reports_which_parent_acts_on_the_scale() -> None:
    topology = [[], [], [0, 1]]
    types = [VariableType.CONTINUOUS] * 3

    dataset = assemble_dataset(
        np.random.default_rng(1), topology, types, LOCAL_SCALE_NOISE_MODELS, 200
    )

    mean_parents, scale_parent = dataset.roles["v2"]
    assert scale_parent in {"v0", "v1"}
    assert sorted([scale_parent, *mean_parents]) == ["v0", "v1"]


@pytest.mark.parametrize("case", list(FundamentalCase))
def test_generated_data_go_through_every_ci_test_after_their_post_processing(
    case: FundamentalCase,
) -> None:
    """Continuous, binary and nominal variables reach all the tests, each with its own
    post-processing, and every test returns a valid p-value.
    """
    topology = fundamental_topology(case, 1)
    types = [VariableType.CONTINUOUS] * len(topology.topology)
    types[topology.y_idx] = VariableType.NOMINAL
    types[topology.z_idxs[0]] = VariableType.BINARY
    data = assemble_dataset(
        np.random.default_rng(2), topology.topology, types, ADDITIVE_NOISE_MODELS, 400
    ).data
    assert not has_constant_column(data)
    x, y, z = f"v{topology.x_idx}", f"v{topology.y_idx}", [f"v{i}" for i in topology.z_idxs]

    methods = [
        (fisher_z_test, to_float_columns),
        (kci_test, to_float_columns),
        (pearson_chi2_test, discretize_continuous_columns),
        (mixed_ci_test, remove_unused_categories),
    ]
    for test, post_process in methods:
        prepared = post_process(data)
        for given in ([], z):
            result = test(prepared, x, y, given)

            assert 0.0 <= result.p_value <= 1.0
