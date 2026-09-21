from collections.abc import Callable
from dataclasses import dataclass

from src.CI_tests.common import CIResult
from src.CI_tests.GCM.gcm import gcm_test
from test.CI_tests.GCM.gcm_r_bridge import gcm_test_r


@dataclass
class RegisteredMethod:
    """A CI test method plugged into the shared cross-language test battery.

    🇬🇧 Pairs a method's Python port with its reference implementation
    (whatever language/library that happens to be — R via rpy2 here,
    could be another Python library for a different method). Adding a
    new method to `METHODS` below is all it takes to run it through
    every scenario in `test_ci_methods.py` — no new test code needed.

    🇪🇸 Empareja el port a Python de un método con su implementación de
    referencia (sea cual sea el lenguaje/librería — aquí R vía rpy2,
    podría ser otra librería Python para otro método). Añadir un método
    nuevo a `METHODS` de abajo es todo lo que hace falta para que pase
    por todos los escenarios en `test_ci_methods.py` — sin test nuevo.
    """

    name: str
    python_impl: Callable[..., CIResult]
    reference_impl: Callable[..., CIResult]


METHODS = [
    RegisteredMethod("gcm", gcm_test, gcm_test_r),
]
