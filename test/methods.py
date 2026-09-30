from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.CI_tests.common import CIResult
from src.CI_tests.Fisher_Z.fisher_z import FisherZ
from src.CI_tests.GCM.gcm import gcm_test
from src.CI_tests.KCI.kci import kci_test
from test.CI_tests.Fisher_Z.fisher_z_causallearn_bridge import fisher_z_test_causallearn
from test.CI_tests.GCM.gcm_r_bridge import gcm_test_r
from test.CI_tests.KCI.kci_causallearn_bridge import kci_test_causallearn


@dataclass
class RegisteredMethod:
    """A CI test method plugged into the shared cross-language test battery.

    🇬🇧 Pairs a method's Python port with its reference implementation
    (R via rpy2 for GCM, causal-learn directly for KCI). `build_kwargs`
    adapts a scenario's raw (x, y, z) into whatever kwargs this method's
    own call signature needs (residuals, raw data, extra config...), so
    every scenario in `SCENARIOS` runs against every method without either
    side knowing about the other's shape. Adding a method here is all it
    takes to run it through `test_ci_methods.py`.

    🇪🇸 Empareja el port a Python de un método con su implementación de
    referencia (R vía rpy2 para GCM, causal-learn directamente para KCI).
    `build_kwargs` adapta el (x, y, z) crudo de un escenario a los kwargs
    que necesite la firma propia de este método (residuos, datos crudos,
    configuración extra...), así todo escenario en `SCENARIOS` corre contra
    todo método sin que ninguno de los dos lados conozca la forma del otro.
    Añadir un método aquí es todo lo que hace falta para que pase por
    `test_ci_methods.py`.
    """

    name: str
    python_impl: Callable[..., CIResult]
    reference_impl: Callable[..., CIResult]
    build_kwargs: Callable[
        [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], dict[str, object]
    ]


def _ols_residuals(v: NDArray[np.float64], z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Closed-form, RNG-free residuals of v on z — decouples the comparison
    from whichever regressor each side would otherwise default to."""
    z_design = np.column_stack([np.ones(z.shape[0]), z])
    coef, *_ = np.linalg.lstsq(z_design, v, rcond=None)
    return v - z_design @ coef


def _gcm_kwargs(
    x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]
) -> dict[str, object]:
    return {"resid_x_on_z": _ols_residuals(x, z), "resid_y_on_z": _ols_residuals(y, z)}


def _kci_kwargs(
    x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]
) -> dict[str, object]:
    return {"data_x": x.reshape(-1, 1), "data_y": y.reshape(-1, 1), "data_z": z}


def _kci_bootstrap_kwargs(
    x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]
) -> dict[str, object]:
    return _kci_kwargs(x, y, z) | {"approx": False}


def _fisher_z_kwargs(
    x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]
) -> dict[str, object]:
    return {"x": x, "y": y, "z": z}


METHODS = [
    RegisteredMethod("gcm", gcm_test, gcm_test_r, _gcm_kwargs),
    RegisteredMethod("kci", kci_test, kci_test_causallearn, _kci_kwargs),
    RegisteredMethod("kci_bootstrap", kci_test, kci_test_causallearn, _kci_bootstrap_kwargs),
    RegisteredMethod("fisher_z", FisherZ().test, fisher_z_test_causallearn, _fisher_z_kwargs),
]
