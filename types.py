#!/usr/bin/env python3

from enum import Enum
from typing import NamedTuple, Optional

class TopicSegmentationDatasets(Enum):
    AMI = 0
    ICSI = 1
    TEST = 2

class TopicSegmentationAlgorithm(Enum):
    RANDOM = 0
    EVEN = 1
    BERT = 2
    SBERT = 3

class TextTilingHyperparameters(NamedTuple):
    SENTENCE_COMPARISON_WINDOW: int = 15
    SMOOTHING_PASSES: int = 2
    SMOOTHING_WINDOW: int = 1
    TOPIC_CHANGE_THRESHOLD: float = 0.6

class TopicSegmentationConfig(NamedTuple):
    TEXT_TILING: Optional[TextTilingHyperparameters] = None
    MAX_SEGMENTS_CAP: bool = True
    MAX_SEGMENTS_CAP__AVERAGE_SEGMENT_LENGTH: int = 60
