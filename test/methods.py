from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import pandas as pd

from src.CI_tests.common import CIResult, frame_to_arrays
from src.CI_tests.Fisher_Z.fisher_z import fisher_z_test
from src.CI_tests.GCM.gcm import GCM
from src.CI_tests.KCI.kci import KCI, kci_test
from src.CI_tests.Mixed_CI.mixed_ci import mixed_ci_test
from src.CI_tests.pearson_chi2.pearson_chi2 import pearson_chi2_test
from src.Experiments.Experimental_cases.common import (
    discretize_continuous_columns,
    to_float_columns,
)
from test.CI_tests.Fisher_Z.fisher_z_causallearn_bridge import fisher_z_test_causallearn
from test.CI_tests.GCM.gcm_r_bridge import gcm_test_r
from test.CI_tests.GCM.ols_regressor import OLSRegressor
from test.CI_tests.KCI.kci_causallearn_bridge import kci_test_causallearn
from test.CI_tests.Mixed_CI.mixed_ci_r_bridge import mixed_ci_test_r
from test.CI_tests.pearson_chi2.pearson_chi2_causallearn_bridge import (
    pearson_chi2_test_causallearn,
)


@dataclass
class RegisteredMethod:
    """A CI test paired with its reference implementation, for the shared parity battery.

    Attributes
    ----------
    name : str
        Identifier of the method.

    python_impl : callable
        The test under check, with the ``CITest`` signature.

    reference_impl : callable
        The reference implementation, with the same signature.

    prepare : callable
        Formatting applied to the DataFrame before both implementations see it,
        so the method can handle every variable type.

    compare_log_p : bool, optional (default: False)
        If True, p-values are compared on the log scale (for tiny p-values).
    """

    name: str
    python_impl: Callable[..., CIResult]
    reference_impl: Callable[..., CIResult]
    prepare: Callable[[pd.DataFrame], pd.DataFrame]
    compare_log_p: bool = False


def _frame_test(array_test: Callable[..., CIResult]) -> Callable[..., CIResult]:
    """Adapt a test taking ``(x, y, z, alpha)`` arrays to the DataFrame signature."""

    def run(
        data: pd.DataFrame, x: str, y: str, z: list[str] | None = None, alpha: float = 0.05
    ) -> CIResult:
        return array_test(*frame_to_arrays(data, x, y, z), alpha)

    return run


METHODS = [
    RegisteredMethod(
        "gcm", _frame_test(GCM(estimator=OLSRegressor()).test), gcm_test_r, to_float_columns
    ),
    RegisteredMethod("kci", kci_test, kci_test_causallearn, to_float_columns),
    RegisteredMethod(
        "kci_bootstrap",
        _frame_test(KCI(approx=False).test),
        partial(kci_test_causallearn, approx=False),
        to_float_columns,
    ),
    RegisteredMethod("fisher_z", fisher_z_test, fisher_z_test_causallearn, to_float_columns),
    RegisteredMethod(
        "pearson_chi2",
        pearson_chi2_test,
        pearson_chi2_test_causallearn,
        discretize_continuous_columns,
    ),
    RegisteredMethod(
        "mixed_ci", mixed_ci_test, mixed_ci_test_r, lambda data: data, compare_log_p=True
    ),
]
