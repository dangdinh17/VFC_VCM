"""Losses package."""

from .distortion import DistortionLoss
from .rate_distortion import RateDistortionLoss

__all__ = ["DistortionLoss", "RateDistortionLoss"]
