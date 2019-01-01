import math
import numpy as np
from scipy import stats
import itertools
import timeseries
import collections

class WeightedMean(object):
	def __init__(self, weight_function, *args, **kwargs):
		r"""
		Argument:
			weight_function (fun), a (start, end) -> (0, inf) function
		"""
		self.f = weight_function
		self.ts = timeseries.TimeSeries(*args, **kwargs)

	def get_procrastination(self, normalize_time=True, normalize_weight=True):
		timestamps = np.array(self.ts.get_timestamps(), dtype=np.float)
		if normalize_time:
			timestamps = timestamps / self.ts.get_duration()

		effort = np.array(self.ts.get_weights(), dtype=np.float)
		if normalize_weight:
			effort = effort / effort.sum()  # normalize

		weight = np.array(map(self.f, timestamps), dtype=np.float)
		return np.dot(weight, effort)
