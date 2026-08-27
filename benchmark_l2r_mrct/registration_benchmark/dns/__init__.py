"""Auditable, versioned reproductions of Mok et al. MASR-Net DNS + IO.

``MASRNet`` is the frozen legacy class used by the protocol-300 checkpoint.
The explicitly versioned ``faithful_v2`` classes are separate so importing the
new implementation cannot silently change an existing experiment.
"""

from .faithful_v2 import (
    DSIRExtraction,
    FAITHFUL_V2_VERSION,
    Figure9FeatureExtractor,
    FullResolutionDSIRExtractor,
    MASRNetFaithfulV2,
    MRCTPreprocessConfig,
    faithful_v2_provenance,
    normalize_mrct_intensity,
    standardize_mrct_volume,
    stochastic_nonlinear_transform_faithful_v2,
)
from .model import MASRNet

__all__ = [
    "DSIRExtraction",
    "FAITHFUL_V2_VERSION",
    "Figure9FeatureExtractor",
    "FullResolutionDSIRExtractor",
    "MASRNet",
    "MASRNetFaithfulV2",
    "MRCTPreprocessConfig",
    "faithful_v2_provenance",
    "normalize_mrct_intensity",
    "standardize_mrct_volume",
    "stochastic_nonlinear_transform_faithful_v2",
]

