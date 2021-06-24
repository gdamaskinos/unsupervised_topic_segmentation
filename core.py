#!/usr/bin/env python3
import baselines as topic_segmentation_baselines
import numpy as np
import pandas as pd
import torch

from transformers import RobertaConfig, RobertaModel
# pretrained roberta model
configuration = RobertaConfig()
roberta_model = RobertaModel(configuration)

from types import (
    TopicSegmentationAlgorithm,
    TopicSegmentationConfig,
    TextTilingHyperparameters,
)

PARALLEL_INFERENCE_INSTANCES = 20

def PrintMessage(msg, x):
    print(msg)
    print(x)

def depth_score(timeseries):
    """
    The depth score corresponds to how strongly the cues for a subtopic changed on both sides of a
    given token-sequence gap and is based on the distance from the peaks on both sides of the valleyto that valley.

    returns depth_scores
    """
    depth_scores = []
    for i in range(1, len(timeseries) - 1):
        left, right = i - 1, i + 1
        while left > 0 and timeseries[left - 1] > timeseries[left]:
            left -= 1
        while (
            right < (len(timeseries) - 1) and timeseries[right + 1] > timeseries[right]
        ):
            right += 1
        depth_scores.append(
            (timeseries[right] - timeseries[i]) + (timeseries[left] - timeseries[i])
        )
    return depth_scores


def smooth(timeseries, n, s):
    smoothed_timeseries = timeseries[:]
    for _ in range(n):
        for index in range(len(smoothed_timeseries)):
            neighbours = smoothed_timeseries[
                max(0, index - s) : min(len(timeseries) - 1, index + s)
            ]
            smoothed_timeseries[index] = sum(neighbours) / len(neighbours)
    return smoothed_timeseries


def sentences_similarity(first_sentence_features, second_sentence_features) -> float:
    """
    Given two senteneces embedding features compute cosine similarity
    """
    similarity_metric = torch.nn.CosineSimilarity()
    return float(similarity_metric(first_sentence_features, second_sentence_features))


def compute_window(timeseries, start_index, end_index):
    """given start and end index of embedding, compute pooled window value

    [window_size, 768] -> [1, 768]
    """
    stack = torch.stack([features[0] for features in timeseries[start_index:end_index]])
    stack = stack.unsqueeze(
        0
    )  # https://jbencook.com/adding-a-dimension-to-a-tensor-in-pytorch/
    stack_size = end_index - start_index
    pooling = torch.nn.MaxPool2d((stack_size - 1, 1))
    return pooling(stack)


def block_comparison_score(timeseries, k):
    """
    comparison score for a gap (i)

    cfr. docstring of block_comparison_score
    """
    res = []
    for i in range(k, len(timeseries) - k):
        first_window_features = compute_window(timeseries, i - k, i + 1)
        second_window_features = compute_window(timeseries, i + 1, i + k + 2)
        res.append(
            sentences_similarity(first_window_features[0], second_window_features[0])
        )

    return res


def get_features_from_sentence(batch_sentences, layer=-2):
    """
    extracts the BERT semantic representation
    from a sentence, using an averaged value of
    the `layer`-th layer

    returns a 1-dimensional tensor of size 758
    """
    batch_features = []
    for sentence in batch_sentences:
        tokens = roberta_model.encode(sentence)
        all_layers = roberta_model.extract_features(tokens, return_all_hiddens=True)
        pooling = torch.nn.AvgPool2d((len(tokens), 1))
        sentence_features = pooling(all_layers[layer])
        batch_features.append(sentence_features[0])
    return batch_features


def arsort2(array1, array2):
    x = np.array(array1)
    y = np.array(array2)

    sorted_idx = x.argsort()[::-1]
    return x[sorted_idx], y[sorted_idx]


def get_local_maxima(array):
    local_maxima_indices = []
    local_maxima_values = []
    for i in range(1, len(array) - 1):
        if array[i - 1] < array[i] and array[i] > array[i + 1]:
            local_maxima_indices.append(i)
            local_maxima_values.append(array[i])
    return local_maxima_indices, local_maxima_values


def depth_score_to_topic_change_indexes(
    depth_score_timeseries,
    meeting_duration,
    topic_segmentation_configs=TopicSegmentationConfig,
):
    """
    capped add a max segment limit so there are not too many segments, used for UI improvements on the Workplace TeamWork product
    """

    capped = topic_segmentation_configs.MAX_SEGMENTS_CAP
    average_segment_length = (
        topic_segmentation_configs.MAX_SEGMENTS_CAP__AVERAGE_SEGMENT_LENGTH
    )
    threshold = topic_segmentation_configs.TEXT_TILING.TOPIC_CHANGE_THRESHOLD * max(
        depth_score_timeseries
    )

    print("DEPTH_SCORE_TIMESERIES:")
    print(list(depth_score_timeseries))

    if depth_score_timeseries == []:
        return []

    local_maxima_indices, local_maxima = get_local_maxima(depth_score_timeseries)

    if local_maxima == []:
        return []

    if capped:  # capped is segmentation used for UI
        # sort based on maxima for pruning
        local_maxima, local_maxima_indices = arsort2(local_maxima, local_maxima_indices)

        # local maxima are sorted by depth_score value and we take only the first K
        # where the K+1th local maxima is lower then the threshold
        for thres in range(len(local_maxima)):
            if local_maxima[thres] <= threshold:
                break

        max_segments = int(meeting_duration / average_segment_length)
        slice_length = min(max_segments, thres)

        local_maxima_indices = local_maxima_indices[:slice_length]
        local_maxima = local_maxima[:slice_length]

        # after pruning, sort again based on indices for chronological ordering
        local_maxima_indices, _ = arsort2(local_maxima_indices, local_maxima)

    else:  # this is the vanilla TextTiling used for Pk optimization
        filtered_local_maxima_indices = []
        filtered_local_maxima = []

        for i, m in enumerate(local_maxima):
            if m > threshold:
                filtered_local_maxima.append(m)
                filtered_local_maxima_indices.append(i)

        local_maxima = filtered_local_maxima
        local_maxima_indices = filtered_local_maxima_indices

    print("LOCAL_MAXIMA_INDICES:")
    print(list(local_maxima_indices))

    return local_maxima_indices


def get_timeseries(caption_indexes, features):
    timeseries = []
    for caption_index in caption_indexes:
        timeseries.append(features[caption_index])
    return timeseries


def flatten_features(batches_features):
    res = []
    for batch_features in batches_features:
        res += batch_features
    return res


def split_list(a, n):
    k, m = divmod(len(a), n)
    return (
        a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(min(len(a), n))
    )


def topic_segmentation(
    topic_segmentation_algorithm: TopicSegmentationAlgorithm,
    df: pd.DataFrame,
    meeting_id_col_name: str,
    start_col_name: str,
    end_col_name: str,
    caption_col_name: str,
    topic_segmentation_config: TopicSegmentationConfig,
):
    """
    Input:
        df: dataframe with meeting captions
    Output:
        {meeting_id: [list of topic change indexes]}
    """

    if topic_segmentation_algorithm == TopicSegmentationAlgorithm.BERT:
        return topic_segmentation_bert(
            df,
            meeting_id_col_name,
            start_col_name,
            end_col_name,
            caption_col_name,
            topic_segmentation_config,
        )
    elif topic_segmentation_algorithm == TopicSegmentationAlgorithm.RANDOM:
        return topic_segmentation_baselines.topic_segmentation_random(
            df, meeting_id_col_name, start_col_name, end_col_name, caption_col_name
        )
    elif topic_segmentation_algorithm == TopicSegmentationAlgorithm.EVEN:
        return topic_segmentation_baselines.topic_segmentation_even(
            df, meeting_id_col_name, start_col_name, end_col_name, caption_col_name
        )
    else:
        raise NotImplementedError("Algorithm not implemented")


def topic_segmentation_bert(
    df: pd.DataFrame,
    meeting_id_col_name: str,
    start_col_name: str,
    end_col_name: str,
    caption_col_name: str,
    topic_segmentation_configs: TopicSegmentationConfig,
):
    textiling_hyperparameters = topic_segmentation_configs.TEXT_TILING

    # parallel inference
    batches_features = []
    for batch_sentences in split_list(
        df[caption_col_name], PARALLEL_INFERENCE_INSTANCES
    ):
        batches_features.append(get_features_from_sentence(batch_sentences))
    features = flatten_features(batches_features)

    # meeting_id -> list of topic change start times
    segments = {}
    task_idx = 0
    print("meeting_id -> task_idx")
    for meeting_id in set(df[meeting_id_col_name]):
        print("%s -> %d" % (meeting_id, task_idx))
        task_idx += 1

        meeting_data = df[df[meeting_id_col_name] == meeting_id]
        caption_indexes = list(meeting_data.index)

        timeseries = get_timeseries(caption_indexes, features)
        block_comparison_score_timeseries = block_comparison_score(
            timeseries, k=textiling_hyperparameters.SENTENCE_COMPARISON_WINDOW
        )

        block_comparison_score_timeseries = smooth(
            block_comparison_score_timeseries,
            n=textiling_hyperparameters.SMOOTHING_PASSES,
            s=textiling_hyperparameters.SMOOTHING_WINDOW,
        )

        depth_score_timeseries = depth_score(block_comparison_score_timeseries)

        meeting_start_time = meeting_data[start_col_name].iloc[0]
        meeting_end_time = meeting_data[end_col_name].iloc[-1]
        meeting_duration = meeting_end_time - meeting_start_time
        segments[meeting_id] = depth_score_to_topic_change_indexes(
            depth_score_timeseries,
            meeting_duration,
            topic_segmentation_configs=topic_segmentation_configs,
        )
        print(segments[meeting_id])

    return segments
