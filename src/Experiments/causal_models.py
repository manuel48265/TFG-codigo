from enum import Enum, auto

from src.Experiments.factory import (
    CategoricalFactory,
    CategoricalVariablesMeanFactory,
    CategoricalVariablesScaleFactory,
    FunctionFactory,
    GaussianNoiseFactory,
    HeteroscedasticShapeFactory,
    HomoscedasticShapeFactory,
    IdentityTransformFactory,
    LinearShapeFactory,
    NoiseFactory,
    NonGaussianNoiseFactory,
    NonLinearShapeFactory,
    OuterTransformFactory,
)


class NodeKind(Enum):
    CONTINUOUS = auto()
    ORDINAL = auto()
    BINARY = auto()
    NOMINAL = auto()


class ModelKind(Enum):
    LINEAR_GAUSSIAN = auto()
    LINEAR_NON_GAUSSIAN = auto()
    ADDITIVE_NOISE_MODELS = auto()
    POST_NON_LINEAR = auto()
    LOCAL_SCALE_NOISE_MODELS = auto()


class Model:
    mean_functions: type[FunctionFactory]
    categorical_mean_functions: type[CategoricalFactory]
    scale_functions: type[FunctionFactory]
    categorical_scale_function: type[CategoricalFactory]
    transform_functions: type[FunctionFactory]
    noise_functions: type[NoiseFactory]

    def __init__(
        self,
        mean_functions: type[FunctionFactory],
        categorical_mean_functions: type[CategoricalFactory],
        scale_functions: type[FunctionFactory],
        categorical_scale_function: type[CategoricalFactory],
        transform_functions: type[FunctionFactory],
        noise_functions: type[NoiseFactory],
    ) -> None:
        self.mean_functions = mean_functions
        self.categorical_mean_functions = categorical_mean_functions
        self.scale_functions = scale_functions
        self.categorical_scale_function = categorical_scale_function
        self.transform_functions = transform_functions
        self.noise_functions = noise_functions


LINEAR_GAUSSIAN = Model(
    LinearShapeFactory,
    CategoricalVariablesMeanFactory,
    HomoscedasticShapeFactory,
    CategoricalVariablesScaleFactory,
    IdentityTransformFactory,
    GaussianNoiseFactory,
)
LINEAR_NON_GAUSSIAN = Model(
    LinearShapeFactory,
    CategoricalVariablesMeanFactory,
    HomoscedasticShapeFactory,
    CategoricalVariablesScaleFactory,
    IdentityTransformFactory,
    NonGaussianNoiseFactory,
)
ADDITIVE_NOISE_MODELS = Model(
    NonLinearShapeFactory,
    CategoricalVariablesMeanFactory,
    HomoscedasticShapeFactory,
    CategoricalVariablesScaleFactory,
    IdentityTransformFactory,
    GaussianNoiseFactory,
)
POST_NON_LINEAR = Model(
    NonLinearShapeFactory,
    CategoricalVariablesMeanFactory,
    HomoscedasticShapeFactory,
    CategoricalVariablesScaleFactory,
    OuterTransformFactory,
    GaussianNoiseFactory,
)
LOCAL_SCALE_NOISE_MODELS = Model(
    NonLinearShapeFactory,
    CategoricalVariablesMeanFactory,
    HeteroscedasticShapeFactory,
    CategoricalVariablesScaleFactory,
    IdentityTransformFactory,
    GaussianNoiseFactory,
)
