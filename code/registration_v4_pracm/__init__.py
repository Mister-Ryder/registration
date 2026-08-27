"""PRA-CM V4: relational probabilistic 3-D correspondence registration."""

from .config import ExperimentConfig, load_config
from .model.pracm_v4 import PRACM3D, RegistrationOutput3D

__all__ = ["ExperimentConfig", "PRACM3D", "RegistrationOutput3D", "load_config"]
__version__ = "4.0.0"
