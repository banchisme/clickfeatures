import pytest
import numpy as np
from clickfeatures import procrastination, timestamps
import random
import itertools
import math
from scipy import stats, fftpack

def test_weighted_mean():
    ts_val = range(7)
    ws1 = [1, 1, 1, 1, 1, 1, 1]
    ws2 = [0, 0, 0, 2, 2, 2, 2]
    ts1 = timestamps.TimeStamps(ts_val, ws1)
    ts2 = timestamps.TimeStamps(ts_val, ws2)
    fun = lambda x: 1. / (1 - x)

    procrastination1 = procrastination.weighted_mean(fun, ts1)
    procrastination2 = procrastination.weighted_mean(fun, ts2)
    assert procrastination1 < procrastination2

