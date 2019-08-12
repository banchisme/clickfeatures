import math
import numpy as np
from scipy import stats
import itertools
from clickfeatures import timestamps
import collections


def weighted_mean(weight_function, ts, normalize_time=True, normalize_effort=True):
    r"""
        return weighted mean as the indicator for procrastination

    :param weight_function (fun): a function that takes in a timestamp and output a weight
    :param ts (timestamps.Timestamps): a timestamps object
    :param normalize_time (bool): if True, timestamps are scaled from (start, end) to (0, 1)
    :param normalize_effort (bool): if True, students effort are normalized
    :return:
        a procrastination index, high value indicates high level of procrastination

    example:
        weight_function = lambda x: x  # larger time is assigned larger weight
        ts = timestamps.TimeStamps([5, 6], [1, 2], end = 6, unit='day')  # a student that works 1 hour on Saturday and 2 hours on Sunday
        procrastination = weighted_mean(weighted_function, ts)
    """

    ts_val = np.array(ts.get_timestamps(), dtype=np.float)

    if normalize_time:
        ts_val = ts_val / ts.get_duration()

    # ts.weight measures the strength of activity at the corresponding timestamp
    # it can be conceptualized as effort
    effort = np.array(ts.get_weights(), dtype=np.float)
    if normalize_effort:
        effort = effort / effort.sum()

    weight = np.array(list(map(weight_function, ts_val)), dtype=np.float)

    return np.dot(weight, effort)  # procrastination as weighted sum of effort
