import numpy as np
import math


def group(timestamps, weights, window, num_groups):
    r"""
        group the timestamps and weights according to window size
        Argument:
            timestamps: timestamps
            weights: weights
            window (int): group size
            num_groups (int): num of groups
        Return:
            time_groups (list of lists), weight_groups (list of lists)
    """
    timestamps = timestamps[:]
    weights = weights[:]
    time_groups = []
    weight_groups = []
    group_end = window

    for _ in range(num_groups):
        time_group = []
        weight_group = []
        while timestamps and timestamps[0] < group_end:
            time_group.append(timestamps.pop(0))
            weight_group.append(weights.pop(0))
        time_groups.append(time_group)
        weight_groups.append(weight_group)
        group_end += window

    return time_groups, weight_groups


def aggregate(timestamps, weights, window, num_groups, agg_fun):
    r"""
        aggregate the weights according to window size
        Argument:
            timestamps: timestamps
            weights: weights
            window (int): group size
            num_groups (int): num of groups
            agg_fun (function): aggregtion function
        Return:
            aggregated weights (list of lists)
    """
    time_groups, weight_groups = group(timestamps, weights, window, num_groups)
    return range(len(time_groups)), map(agg_fun, weight_groups)


class TimeSeries(object):
    def __init__(self, ts, weigths=None, start=0, end=None, unit='minute'):
        r"""
            time is stored based on the unit
        """
        assert (
            unit in ['second', 'minute', 'hour', 'day', 'week'],
            'not supported unit')
        self.unit = unit
        self.ts = list(ts)
        self.weights = list(weigths or [1] * len(ts))
        self.start = int(math.floor(start))
        self.end = int((end or math.ceil(max(ts))))

    def get_timestamps(self):
        r"""
        Return:
            timestamps (np.array), raw timestamps provided during initialization
        """
        return self.ts

    def get_weights(self):
        r"""
        return the weights vector for the corresponding time
        """
        return self.weights

    def get_unit(self):
        r"""
        Return:
            unit (string), base unit of the timestamps
        """
        return self.unit

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

    def get_duration(self):
        r"""
        Return:
            time duration
        """
        return self.end - self.start + 1

    def quantify(self, time_unit):
        r"""given a time_unit, quantify it a the scale of the base unit
        Argument:
            time_unit (string): time unit, e.g., "hour"
        Return:
            quantity (int): quantity in the scale of the base unit
        """
        size_dictionary = {
            'second': 1,
            'minute': 60,
            'hour': 60 * 60,
            'day': 60 * 60 * 24,
            'week': 60 * 60 * 24 * 7}

        base_unit = size_dictionary[self.get_unit()]
        size_unit = size_dictionary[time_unit]
        if size_unit / base_unit >= 1:
            return size_unit / base_unit
        else:
            raise ValueError(
                'time unit "{}"" is smaller than the base unit: "{}"'.
                format(time_unit, self.get_unit()))

    def active(self, t, window):
        r"""
        check if the user is active in time t, based on the window size
        Argument:
            t (float), timestamps, in base unit
            window (int)
        Return:
            1 if True, 0, o.w.
        """

        timestamps = self.get_timestamps()[:]
        active_indexes = set([int(timestamp // window) for timestamp in timestamps])
        index = int(t // window)
        return int(index in active_indexes)

    def group(self, window):
        r"""
        group the timestamps and weights according to window size
        Argument:
            weights: weights
            window (int): group size
        Return:
            time_groups (list of lists), weight_groups (list of lists)
        """
        duration = self.get_duration()
        num_groups = int(math.ceil(1. * duration / window))
        timestamps = self.get_timestamps()[:]
        weights = self.get_weights()[:]
        return group(timestamps, weights, window, num_groups)

    def aggregate(self, window, agg_fun):
        r"""
        aggregate the weights according to window size
        Argument:
            window (int): group size
            agg_fun (function): aggregtion function
        Return:
            aggregated weights (list of lists)
        """
        duration = self.get_duration()
        num_groups = int(math.ceil(1. * duration / window))
        timestamps = self.get_timestamps()[:]
        weights = self.get_weights()[:]
        return aggregate(timestamps, weights, window, num_groups, agg_fun)
