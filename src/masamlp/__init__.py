"""masaMLP: extensible tabular deep learning.

TabularResNet, DANet, and TabularLNN behind sklearn-compatible estimators
with first-class sample_weight, custom objectives, custom metrics, and early
stopping on any metric — the sibling library of repleafgbm.
"""

from masamlp.classifier import MasaClassifier
from masamlp.core.metrics import BaseMetric, get_metric, make_metric
from masamlp.core.objectives import (
    BaseObjective,
    BinaryLogistic,
    Huber,
    MulticlassSoftmax,
    PoissonRegression,
    Quantile,
    get_objective,
    make_objective,
)
from masamlp.models import register_model
from masamlp.presets import realmlp_params, realmlp_td_params
from masamlp.regressor import MasaRegressor

__version__ = "0.3.0"

__all__ = [
    "MasaRegressor",
    "MasaClassifier",
    "BaseMetric",
    "get_metric",
    "make_metric",
    "BaseObjective",
    "get_objective",
    "make_objective",
    "Huber",
    "Quantile",
    "PoissonRegression",
    "BinaryLogistic",
    "MulticlassSoftmax",
    "register_model",
    "realmlp_params",
    "realmlp_td_params",
    "__version__",
]
