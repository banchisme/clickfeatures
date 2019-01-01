import pytest
import numpy as np
from clickfeatures import procrastination
import random
import itertools
import math
from scipy import stats, fftpack


def test_weighted_mean():
	ts = range(7)
	ws1 = [1, 1, 1, 1, 1, 1, 1]
	ws2 = [0, 0, 0, 2, 2, 2, 2]
	fun = lambda x: 1. / (1 - x)
	obj1 = procrastination.WeightedMean(fun, ts, ws1)
	obj2 = procrastination.WeightedMean(fun, ts, ws2)
	assert obj1.get_procrastination() < obj2.get_procrastination()
