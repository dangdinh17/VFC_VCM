"""Datasets package."""

from .test_dataset import TestVideoFeatureDataset, test_video_feature_collate_fn
from .train_dataset import TrainVideoFeatureDataset, train_video_feature_collate_fn
from .semantic_segmentation import SemanticSegmentationDataset

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
	"SemanticSegmentationDataset",
	"train_video_feature_collate_fn",
	"test_video_feature_collate_fn",
]
