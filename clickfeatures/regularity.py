import math
import numpy as np
from scipy import stats
import itertools
import timeseries
import collections


class TimeRegularity(object):
    def __init__(self, *args, **kwargs):
        r"""
            time is stored based on the unit
        """
        self.ts = timeseries.TimeSeries(*args, **kwargs)

    def _stack(self, window, windows_per_layer, dtype=np.float64):
        r"""
        Argument:
            window (int): window in base unit
            windows_per_layer (int): number of windows in each layer
        Return:
            np.array, the number of layer that a window is active

        For example, base_unit = 1 day, window = 1, windows_per_layer = 7
        then, this function returns "number of weeks the user is active on each day"
        """
        ts, ws = self.ts.aggregate(
            window,
            lambda group: int(sum(group) > 0))

        res = [0] * windows_per_layer
        for t in ts:
            res[t % windows_per_layer] += ws[t]
        return np.array(res, dtype=dtype)

    def _week_profile(self):
        r"""
        week_profile = [P(1), P(2), ..., P(Lw)],
        Lw is the number of weeks in the time series
        P(k) = [P(1, k), P(2, k), ..., P(7, k)]
        P(d, k) represents the number of hours user was active in day d of week k"""
        if self.ts.get_unit() in ('second', 'minute', 'hour'):
            window = self.ts.quantify('hour')
            ts, ws = self.ts.aggregate(
                window,
                lambda group: int(sum(group) > 0))
            num_groups = int(math.ceil(
                float(self.ts.get_duration()) / self.ts.quantify('day')))
            ts, ws = timeseries.aggregate(ts, ws, 24, num_groups, sum)
        elif self.ts.get_unit() == 'day':
            ts, ws = self.ts.aggregate(1, sum)
        else:
            raise ValueError('Time series is not profilable.')

        # reshape the profile_vector to matrix
        profile_matrix = []
        pt = 0
        while pt < len(ws):
            col = ws[pt: pt + 7]
            col.extend([0] * (7 - len(col)))
            profile_matrix.append(np.array(col, dtype=np.float64))
            pt += 7

        return profile_matrix

    def _active_days(self, p):
        r"""
            Argument:
                p (np.array) week profile
            Return:
                a set of active days in the profile
        """
        res = set([])
        for day, num_active_hours in enumerate(p):
            if num_active_hours > 0:
                res.add(day)
        return res

    def _scaled_entropy(self, pmf, shift, stretch, base=2):
        entropy = stats.entropy(pmf, base=base)
        return (math.log(shift, base) - entropy) * stretch

    def _sim1(self, p1, p2):
        r"""
        similarity between two profiles p1 and p2,
        based on the number of same active days,
        normalize to have max(sim1) == 1
        """
        act1 = self._active_days(p1)
        act2 = self._active_days(p2)
        res = 1. * len(act1.intersection(act2)) / max(len(act1), len(act2), 1)
        return res

    def _sim2(self, p1, p2, base=2):
        def jsd(p1, p2, base=2):
            # p1 and p2 must be normalized
            return (
                stats.entropy(p1 / 2 + p2 / 2, base=base) -
                stats.entropy(p1, base=base) / 2 -
                stats.entropy(p2, base=base) / 2)

        if np.count_nonzero(p1) > 0 and np.count_nonzero(p2) > 0:
            p1 = p1 / p1.sum()
            p2 = p2 / p2.sum()
            return 1 - jsd(p1, p2) / math.log(base, 2)
        else:
            return np.NaN

    def _sim3(self, p1, p2):
        act1 = self._active_days(p1)
        act2 = self._active_days(p2)
        scale = 1. / max(len(act1.union(act2)), 1)
        numerator = p1 - p2
        denominator = np.array(map(lambda x: float(max(x, 1)), (p1 + p2)))
        res = ((numerator / denominator) ** 2).sum()

        return 1 - scale * res

    def _ws(self, sim_fun):
        r"""ws is short for week similarity
            Arguments:
                sim_fun (function): a function that return similarity between two weeks
            Return:
                averaged pairwise similarity btween all week pairs
        """
        profile = self._week_profile()
        res = 0
        size = 0
        for p1, p2 in itertools.combinations(profile, 2):
            sim = sim_fun(p1, p2)
            if sim >= 0:  # exclude corner cases where sim = None or np.nan
                res += sim
                size += 1
        if size == 0:
            print(profile)
            print(self.ts.get_timestamps())
        return 1. * res / size

    def _fft(self, freq, signal, sample_space):
        r"""give the signal magnitude based on fourier transformation
        Arguments:
            freq (float): frequency of interest
            signal (iterable): singal
            sample_space (int): the time between two sampling points, in minutes
        Returns:
            the magnitude of sub-signal of freqency "freq"
        """

        res = 0j
        for k, s in enumerate(signal):
            res += s * (math.e ** (-2 * math.pi * freq * 1j * k * sample_space))
        return abs(res)

    def _get_signal(self, window):
        r"""
            Argument:
                window (int): window size
            Return:
                signal (np.array): [0, 1, ...]. 1 if self.ts is active in the window
        """
        _, singal = self.ts.aggregate(
            window,
            lambda group: int(sum(group) > 0))
        return np.array(singal, dtype=np.float64)

    def _freq_based_reg(self, freq, window):
        signal = self._get_signal(window)
        return self._fft(freq, signal, window)

    def _time_based_reg(self, window, windows_per_layer):
        histgram = self._stack(window, windows_per_layer)
        # prepare arguments
        pmf = histgram / histgram.sum()
        shift = windows_per_layer
        stretch = histgram.max()
        return self._scaled_entropy(pmf, shift, stretch)

    def pdh(self):
        window = self.ts.quantify('hour')
        layer = self.ts.quantify('day')
        window_per_layer = layer / window
        return self._time_based_reg(window, window_per_layer)

    def pwd(self):
        window = self.ts.quantify('day')
        layer = self.ts.quantify('week')
        window_per_layer = layer / window
        return self._time_based_reg(window, window_per_layer)

    def ws1(self):
        return self._ws(self._sim1)

    def ws2(self):
        return self._ws(self._sim2)

    def ws3(self):
        return self._ws(self._sim3)

    def fdh(self):
        freq = 1. / self.ts.quantify('day')
        window = self.ts.quantify('hour')
        return self._freq_based_reg(freq, window)

    def fwh(self):
        freq = 1. / self.ts.quantify('week')
        window = self.ts.quantify('hour')
        return self._freq_based_reg(freq, window)

    def fwd(self):
        freq = 1. / self.ts.quantify('week')
        window = self.ts.quantify('day')
        return self._freq_based_reg(freq, window)

    def get_regularity_indicators(self):
        granularity = [
            ('hour', ['pdh', 'fdh', 'fwh']),
            ('day', ['pwd', 'ws1', 'ws2', 'ws3', 'fwd'])]
        length = [
            ('week', ['pwd', 'ws1', 'ws2', 'ws3', 'fwh', 'fwd']),
            ('day', ['pdh', 'fdh'])]

        gset = set([])
        lset = set([])

        for unit, indicators in granularity:
            try:
                self.ts.quantify(unit)
            except ValueError:
                continue
            else:
                gset = gset.union(set(indicators))

        for unit, indicators in length:
            try:
                if self.ts.get_duration() / self.ts.quantify(unit) >= 2:
                    lset = lset.union(set(indicators))
            except ValueError:
                continue

        return sorted(list(gset.intersection(lset)))

    def get_regularity(self):
        indicators = self.get_regularity_indicators()
        res = {}
        for indicator in indicators:
            res[indicator] = getattr(self, indicator)()
        return res


class ActivityRegularity(object):
    def __init__(self, activity_names, activities):
        r"""
        Argument:
            activity_names (list): name of activities
            activities (list of lists): each inside list represents the weights
                of activities at a timestamp.
        Return:
            None

        Example:
            a student has:
            50 forum activity in day 1, 40 forum activity in day 2,
            30 content activity in day 1, 0 content activity in day 2
            ar = ActivityRegularity(['forum', 'content'], [[50, 30], [40, 0]])
        """
        # change acts into a list of pmf
        self.activity_names = activity_names
        self.activities = np.array(activities, dtype=np.float64)

    def jsd(self, weighted=False, base=2):
        r"""
        Argument:
            weighted (bool, optional): wether weight pmf by a weight
                Default::False
            base (float or int): base of entropy
        Return:
            Jensen-Shannon divergence among pmfs
        """

        if weighted:
            weights = self.activities.sum(axis=1) / self.activities.sum()
        else:
            num_rows = self.activities.shape[0]
            weights = np.full(num_rows, 1. / num_rows)

        pmfs = self.activities / self.activities.sum(axis=1).reshape((-1, 1))
        weighted_pmfs = np.dot(weights, pmfs)
        pmf_entropies = np.apply_along_axis(
            stats.entropy, 1, pmfs, base=base)
        divergence = (
            stats.entropy(weighted_pmfs, base=base) -
            np.dot(weights, pmf_entropies))

        return divergence

    def concentration(self, base=2):
        r"""
        Argument:
            base (numeric, optional): base for entropy, Default::2
        Return
            entropy of stacked daily: activity
        """
        stack = self.activities.sum(axis=0) / self.activities.sum()
        return 1 - (
            stats.entropy(stack, base=base) /
            stats.entropy([1] * stack.shape[0], base=base))

    def get_regularity(self, weighted=True, base=2):
        r"""
        Argument:
            weigthed (bool, optional): decide if jsd between pmfs are weighted.
                Default:: True
            base (numerical, optional): base for stats.entropy
        Return:
            activity regularity
        """
        return {
            'consitency': 1 - self.jsd(weighted=weighted, base=base),
            'concentration': self.concentration(base=base)}
