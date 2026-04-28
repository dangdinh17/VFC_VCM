"""Datasets package."""

from .test_dataset import TestVideoFeatureDataset, test_video_feature_collate_fn
from .train_dataset import TrainVideoFeatureDataset, train_video_feature_collate_fn
from .vos2019_segmentation import VOS2019BinarySegmentationDataset

try:
	from .single_dataset import (
		TestSequenceDataset,
		TestSingleDataset,
		TrainDoubleDataset,
		TrainSingleDataset,
	)
except Exception:  # optional dependency path (e.g. detectron2 not installed)
	TestSequenceDataset = None
	TestSingleDataset = None
	TrainDoubleDataset = None
	TrainSingleDataset = None

__all__ = [
	"TrainSingleDataset",
	"TestSingleDataset",
	"TrainDoubleDataset",
	"TestSequenceDataset",
	"TrainVideoFeatureDataset",
	"TestVideoFeatureDataset",
	"VOS2019BinarySegmentationDataset",
	"train_video_feature_collate_fn",
	"test_video_feature_collate_fn",
]
