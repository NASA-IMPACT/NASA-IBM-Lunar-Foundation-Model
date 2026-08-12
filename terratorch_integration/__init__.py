"""Custom TerraTorch modules for lunar-fm models."""

__version__ = "2.0.0"

# Import registration modules to trigger @register decorators
# This happens when TerraTorch imports the terratorch_integration package
from . import lunar_register
from .data_adapter import LunarCraterDataModule

# Explicitly import the registration functions to ensure they're executed
from .lunar_register import (
    ni_lfm_v1_tiny,
    ni_lfm_v1_base,
    ni_lfm_v1_large,
)
from .necks import LearnedTokenProjection, SimpleFeaturePyramid, MultilayerSimpleFeaturePyramid
from .decoders import SumFuseDeepGNDecoder
from .lunar_object_detection_task import LunarObjectDetectionTask
from .lunar_segmentation_task import (
    LunarSegmentationTask,
    LunarShapeSegmentationTask,
)
from .lunar_regression_task import LunarPixelwiseRegressionTask
from .lunar_classification_task import (
    LunarClassificationTask,
    LunarScalarRegressionTask,
)
