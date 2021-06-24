#!/usr/bin/env python3
import logging
from bisect import bisect
from typing import Dict

from core import topic_segmentation
from dataset import (
    ami_dataset,
    icsi_dataset,
)
from types import (
    TopicSegmentationAlgorithm,
    TopicSegmentationDatasets,
    TopicSegmentationConfig,
)
from nltk.metrics.segmentation import pk, windowdiff


def compute_metrics(prediction_segmentations, binary_labels, metric_name_suffix=""):
    print(prediction_segmentations)
    print(binary_labels)
    _pk, _windiff = [], []
    for meeting_id, reference_segmentation in binary_labels.items():

        predicted_segmentation_indexes = prediction_segmentations[meeting_id]
        # we need to convert from topic changes indexes to topic changes binaries
        predicted_segmentation = [0] * len(reference_segmentation)
        for topic_change_index in predicted_segmentation_indexes:
            predicted_segmentation[topic_change_index] = 1

        reference_segmentation = "".join(map(str, reference_segmentation))
        predicted_segmentation = "".join(map(str, predicted_segmentation))

        _pk.append(pk(reference_segmentation, predicted_segmentation))

        # setting k to default value used in CoAP (pk) function for both evaluation functions
        k = int(
            round(
                len(reference_segmentation) / (reference_segmentation.count("1") * 2.0)
            )
        )
        _windiff.append(windowdiff(reference_segmentation, predicted_segmentation, k))

    avg_pk = sum(_pk) / len(binary_labels)
    avg_windiff = sum(_windiff) / len(binary_labels)

    print("Pk on {} meetings: {}".format(len(binary_labels), avg_pk))
    print("WinDiff on {} meetings: {}".format(len(binary_labels), avg_windiff))

    return {
        "average_Pk_" + str(metric_name_suffix): avg_pk,
        "average_windiff_" + str(metric_name_suffix): avg_windiff,
    }


def binary_labels_flattened(
    input_df,
    labels_df,
    meeting_id_col_name: str,
    start_col_name: str,
    end_col_name: str,
    caption_col_name: str,
):
    """
    Binary Label [0, 0, 1, 0] for topic changes as ntlk format.
    Hierarchical topic strutcure flattened.
    see https://www.XXXX.com/intern/anp/view/?id=434543
    """
    labels_flattened = {}
    meeting_ids = list(set(input_df[meeting_id_col_name]))

    for meeting_id in meeting_ids:
        logging.info("\n\nMEETING ID:{}".format(meeting_id))

        if meeting_id not in list(labels_df[meeting_id_col_name]):
            logging.info("{} not found in `labels_df`".format(meeting_id))
            continue

        meeting_data = input_df[
            input_df[meeting_id_col_name] == meeting_id
        ].sort_values(by=[start_col_name])
        meeting_sentences = [*map(lambda s: s.lower(), list(meeting_data["caption"]))]

        caption_start_times = list(meeting_data[start_col_name])
        segment_start_times = list(
            labels_df[labels_df[meeting_id_col_name] == meeting_id][start_col_name]
        )

        meeting_labels_flattened = [0] * len(caption_start_times)

        # we skip first and last labaled segment cause they are naive segments
        for sst in segment_start_times[1:]:
            try:
                topic_change_index = caption_start_times.index(sst)
            except ValueError:
                topic_change_index = bisect(caption_start_times, sst)
                if topic_change_index == len(meeting_labels_flattened):
                    topic_change_index -= 1  # bisect my go out of boundary
            meeting_labels_flattened[topic_change_index] = 1

        labels_flattened[meeting_id] = meeting_labels_flattened

        logging.info("MEETING TRANSCRIPTS")
        for i, sentence in enumerate(meeting_sentences):
            if meeting_labels_flattened[i] == 1:
                logging.warning("\n\n<<------ Topic Change () ------>>\n")
            logging.info(sentence)

    return labels_flattened


def binary_labels_top_level(
    input_df,
    labels_df,
    meeting_id_col_name: str,
    start_col_name: str,
    end_col_name: str,
    caption_col_name: str,
):
    """
    Binary Label [0, 0, 1, 0] for topic changes as ntlk format.
    Hierarchical topic strutcure only top level topics
    see https://www.XXXX.com/intern/anp/view/?id=434543
    """
    labels_top_level = {}
    meeting_ids = list(set(input_df[meeting_id_col_name]))

    for meeting_id in meeting_ids:
        logging.info("\n\nMEETING ID:{}".format(meeting_id))

        if meeting_id not in list(labels_df[meeting_id_col_name]):
            logging.info("{} not found in `labels_df`".format(meeting_id))
            continue

        meeting_data = input_df[
            input_df[meeting_id_col_name] == meeting_id
        ].sort_values(by=[start_col_name])
        meeting_sentences = [*map(lambda s: s.lower(), list(meeting_data["caption"]))]

        caption_start_times = list(meeting_data[start_col_name])
        segment_start_times = list(
            labels_df[labels_df[meeting_id_col_name] == meeting_id][start_col_name]
        )
        segment_end_times = list(
            labels_df[labels_df[meeting_id_col_name] == meeting_id][end_col_name]
        )

        meeting_labels_top_level = [0] * len(caption_start_times)

        high_level_topics_indexes = []
        i = 0
        while i < len(segment_end_times):
            end = segment_end_times[i]
            high_level_topics_indexes.append(i)
            if segment_end_times.count(end) == 2:
                # skip all the subtopics of this high level topic
                i = (
                    segment_end_times.index(end)
                    + segment_end_times[segment_end_times.index(end) + 1 :].index(end)
                    + 2
                )
            else:
                i += 1

        segment_start_times_high_level = [
            segment_start_times[i] for i in high_level_topics_indexes
        ]

        # we skip first and last labaled segment cause they are naive segments
        for sst in segment_start_times_high_level[1:]:
            try:
                topic_change_index = caption_start_times.index(sst)
            except ValueError:
                topic_change_index = bisect(caption_start_times, sst)
                if topic_change_index == len(meeting_labels_top_level):
                    topic_change_index -= 1  # bisect my go out of boundary
            meeting_labels_top_level[topic_change_index] = 1

        labels_top_level[meeting_id] = meeting_labels_top_level

        logging.info("MEETING TRANSCRIPTS")
        for i, sentence in enumerate(meeting_sentences):
            if meeting_labels_top_level[i] == 1:
                logging.warning("\n\n<<------ Topic Change () ------>>\n")
            logging.info(sentence)

    return labels_top_level


MEETING_ID_COL_NAME = "meeting_id"
START_COL_NAME = "st"
EN_COL_NAME = "en"
CAPTION_COL_NAME = "caption"


def eval_topic_segmentation(
    dataset_name: TopicSegmentationDatasets,
    topic_segmentation_algorithm: TopicSegmentationAlgorithm,
    topic_segmentation_config: TopicSegmentationConfig,
) -> Dict[str, float]:

    if dataset_name == TopicSegmentationDatasets.AMI:
        input_df, label_df = ami_dataset()
    elif dataset_name == TopicSegmentationDatasets.ICSI:
        input_df, label_df = icsi_dataset()
    elif dataset_name == TopicSegmentationDatasets.TEST:
        input_df, label_df = test_video_dataset()
    else:
        raise NotImplementedError("Unknown dataset_name given.")

    prediction_segmentations = topic_segmentation(
        topic_segmentation_algorithm,
        input_df,
        MEETING_ID_COL_NAME,
        START_COL_NAME,
        EN_COL_NAME,
        CAPTION_COL_NAME,
        topic_segmentation_config,
    )

    flattened = binary_labels_flattened(
        input_df,
        label_df,
        MEETING_ID_COL_NAME,
        START_COL_NAME,
        EN_COL_NAME,
        CAPTION_COL_NAME,
    )

    top_level = binary_labels_top_level(
        input_df,
        label_df,
        MEETING_ID_COL_NAME,
        START_COL_NAME,
        EN_COL_NAME,
        CAPTION_COL_NAME,
    )

    flattened_metrics = compute_metrics(
        prediction_segmentations, flattened, metric_name_suffix="flattened"
    )
    top_level_metrics = compute_metrics(
        prediction_segmentations, top_level, metric_name_suffix="top_level"
    )

    def merge_metrics(*metrics):
        res = {}
        for m in metrics:
            for k, v in m.items():
                res[k] = v
        return res

    return merge_metrics(flattened_metrics, top_level_metrics)
