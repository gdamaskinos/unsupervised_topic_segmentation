from random import random

import pandas as pd


def topic_segmentation_random(
    df: pd.DataFrame,
    meeting_id_col_name: str,
    start_col_name: str,
    end_col_name: str,
    caption_col_name: str,
    random_threshold: float = 0.9,
):

    # meeting_id -> list of topic change start times
    segments = {}
    task_idx = 0
    print("meeting_id -> task_idx")
    for meeting_id in set(df[meeting_id_col_name]):
        print("%s -> %d" % (meeting_id, task_idx))
        task_idx += 1

        meeting_data = df[df[meeting_id_col_name] == meeting_id]
        meeting_start_times = meeting_data[start_col_name]
        random_segmentation = []
        for i, _ in enumerate(meeting_start_times):
            if random() > random_threshold:
                random_segmentation.append(i)
        print(random_segmentation)
        segments[meeting_id] = random_segmentation
    return segments


def topic_segmentation_even(
    df: pd.DataFrame,
    meeting_id_col_name: str,
    start_col_name: str,
    end_col_name: str,
    caption_col_name: str,
):

    # meeting_id -> list of topic change start times
    segments = {}
    task_idx = 0
    print("meeting_id -> task_idx")
    for meeting_id in set(df[meeting_id_col_name]):
        print("%s -> %d" % (meeting_id, task_idx))
        task_idx += 1

        meeting_data = df[df[meeting_id_col_name] == meeting_id]
        meeting_start_times = meeting_data[start_col_name]
        even_segmentation = []
        for i, _ in enumerate(meeting_start_times):
            if i % 30 == 0:
                even_segmentation.append(i)
        print(even_segmentation)
        segments[meeting_id] = even_segmentation
    return segments
