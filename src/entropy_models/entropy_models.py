"""Thin wrapper helpers that expose entropy model constructors from compressai.

These helpers make it easy to import and instantiate the common entropy
components used in the project. They simply forward arguments to the
compressai implementations.
"""

from compressai.entropy_models import (
	EntropyBottleneck,
	EntropyModel,
	GaussianConditional,
)

__all__ = [
	"EntropyBottleneck",
	"EntropyModel",
	"GaussianConditional",

]