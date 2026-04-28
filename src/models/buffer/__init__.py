"""Buffer package for frame feature buffers.

This package exposes `FrameFeatureBuffer`, a small fixed-size FILO
buffer implementation for storing features from previous frames.
"""

from .buffer import FeatureBuffer

__all__ = ["FeatureBuffer"]

