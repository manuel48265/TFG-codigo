from src.Experiments.Node_Generation.factory import (
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


class Model:
    """Set of factories that defines a family of structural causal models.

    Each node of a generated graph draws its mean, scale, transform and noise
    from these factories, so the family fixes what kind of functional
    dependence and noise the data can have.

    Parameters
    ----------
    mean_functions : type of FunctionFactory
        Factory of the mean function of the continuous parents.

    categorical_mean_functions : type of CategoricalFactory
        Factory of the mean contribution of the categorical parents.

    scale_functions : type of FunctionFactory
        Factory of the standard-deviation function of the continuous parents.

    categorical_scale_function : type of CategoricalFactory
        Factory of the standard-deviation factor of the categorical parents.

    transform_functions : type of FunctionFactory
        Factory of the outer transform applied to the noisy mean.

    noise_functions : type of NoiseFactory
        Factory of the noise sampler.
    """

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


# Model families. All use Gaussian noise except the linear non-Gaussian one (Laplace).
# - LINEAR_GAUSSIAN: linear means and homoscedastic noise; exactly linear, so a partial
#   correlation test is exact on it.
# - LINEAR_NON_GAUSSIAN: the same with Laplace noise.
# - ADDITIVE_NOISE_MODELS: nonlinear means, additive homoscedastic noise.
# - POST_NON_LINEAR: nonlinear means and an increasing outer transform; the model is not
#   additive but the dependences, and their detectability, are kept.
# - LOCAL_SCALE_NOISE_MODELS: nonlinear means and a scale parent that changes the noise
#   level inside the detectable range.
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
