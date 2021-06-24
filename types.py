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

class TopicSegmentationConfig(NamedTuple):
    pass
