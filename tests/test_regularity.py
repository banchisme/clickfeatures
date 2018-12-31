import pytest
import numpy as np
from clickfeatures import regularity
import random
import itertools
import math
from scipy import stats, fftpack


@pytest.fixture
def reg_obj():
    length = 60 * 24
    ts = np.array(sorted(random.sample(range(length), length / 2)))
    obj = regularity.Regularity(ts, end=length - 1)
    return obj


@pytest.fixture
def null_obj():
    return regularity.Regularity([0])


@pytest.fixture
def wave_objs():
    f1 = 1. / 60 / 24  # period = 1 day
    f2 = 1. / 60 / 24 / 7  # period = 1 week

    N = 60 * 24 * 7 * 4  # 1 month
    T = 1.  # 1 minute
    xt1 = sin_waves(10, f1, N, T)
    xt2 = sin_waves(10, f2, N * 7, T)
    ts1 = create_timestamps(xt1)
    ts2 = create_timestamps(xt2)
    obj1 = regularity.Regularity(ts1, end=N - 1)
    obj2 = regularity.Regularity(ts2, end=N * 7 - 1)

    return [(f1, obj1), (f2, obj2)]


def sin_waves(amplitude, frequency, N, T):
    t = np.linspace(0.0, N * T, N)
    wave = amplitude * np.sin(frequency * 2. * math.pi * t)

    return wave


def create_timestamps(xt, pct=80):
    threshold = np.percentile(xt, pct)
    return filter(lambda i: xt[i] > threshold, range(len(xt)))


def test_pdh_pwd():
    size = 24 * 7
    obj = regularity.Regularity(np.array(range(size)) * 60, end=size * 60 - 1)

    assert obj.pdh() == pytest.approx(
        7 * (math.log(24, 2) - stats.entropy(np.ones(24) / 24., base=2)), 0.001)
    assert obj.pwd() == pytest.approx(
        math.log(7, 2) - stats.entropy(np.ones(7) / 7., base=2), 0.001)


def test_profile():
    r"""
        this tests given an active user that only works on every hour,
        the profile function returns a matrix of 24s
    """
    size = 24 * 7 * 7
    obj = regularity.Regularity(np.array(range(size)) * 60, end=size * 60 - 1)

    res = obj._week_profile()
    for col in res:
        assert np.allclose(np.array([24] * 7), col)



def test_active_days():
    p = np.array([1, 2, 5, 0, 0, 0, 0])
    obj = regularity.Regularity([0])
    assert obj._active_days(p) == set([0, 1, 2])


def test_sim1(null_obj):
    p1 = np.array([1, 5, 7, 4, 9, 0, 0])
    p2 = np.array([1, 5, 6, 7, 9, 0, 0])
    p3 = np.array([1, 0, 0, 0, 0, 0, 0])
    p4 = np.array([1, 0, 3, 0, 5, 0, 7])

    res1 = null_obj._sim1(p1, p2)
    res2 = null_obj._sim1(p1, p3)
    res3 = null_obj._sim1(p1, p4)
    res4 = null_obj._sim1(p2, p3)
    res5 = null_obj._sim1(p2, p4)
    res6 = null_obj._sim1(p3, p4)

    assert res1 == 1
    assert res2 == 0.2
    assert res3 == 0.6
    assert res4 == 0.2
    assert res5 == 0.6
    assert res6 == 0.25


def test_sim2(null_obj):
    p1 = np.array([0.25, 0.25, 0.25, 0.25])
    p2 = np.array([0.24, 0.26, 0.25, 0.25])
    p3 = np.array([0.22, 0.28, 0.22, 0.28])

    assert null_obj._sim2(p1, p2) > null_obj._sim2(p1, p3)
    assert null_obj._sim2(p1, p1) == 1


def test_sim3(null_obj):
    p1 = np.array([0.25, 0.25, 0.25, 0.25])
    p2 = np.array([0.24, 0.26, 0.25, 0.25])
    p3 = np.array([0.22, 0.28, 0.22, 0.28])

    assert null_obj._sim3(p1, p2) > null_obj._sim3(p1, p3)
    assert null_obj._sim3(p1, p1) == 1


def test_ws():
    size = 24 * 7 * 7
    t1 = np.array(range(size))
    t2 = np.array(sorted(random.sample(range(size), size / 10)))

    obj1 = regularity.Regularity(t1 * 60, end=size * 60 - 1)
    obj2 = regularity.Regularity(t2 * 60, end=size * 60 - 1)

    assert obj1.ws1() == obj1.ws2() == obj1.ws3() == 1
    assert obj1.ws1() > obj2.ws1()
    assert obj1.ws2() > obj2.ws2()
    assert obj1.ws3() > obj2.ws3()


def test_fft(null_obj):
    N = 600
    T = 1.0 / 800.0
    x = np.linspace(0.0, N * T, N)
    y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x)
    yf = fftpack.fft(y)
    xf = fftpack.fftfreq(N, T)

    check_size = 10
    for i, freq in enumerate(xf[:check_size]):
        expect = np.abs(yf[i])
        result = null_obj._fft(freq, y, T)
        assert np.isclose(expect, result)


def test_get_signal():
    ts = np.array([1, 3, 5, 12, 65, 77, 89, 199, 1088])
    obj = regularity.Regularity(ts)
    result = obj._get_signal(60)
    expect = np.array(
        [1., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 1.])
    assert np.array_equal(result, expect)


def test_freq_based_reg():
    r"""check if the fft value at the true frequency is a local maximum"""
    f = 1. / 60 / 24 / 7
    a = 10
    N = 60 * 24 * 7 * 4 * 6 # 6 month
    T = 1.  # 1 minute
    xt = sin_waves(a, f, N, T)
    ts = create_timestamps(xt)
    window = 60
    obj = regularity.Regularity(ts, end=N - 1)
    nearby = [f + i * f / 15 for i in range(-10, 10) if i != 0]
    nearby_fft = [obj._freq_based_reg(freq, window) for freq in nearby]
    assert obj._freq_based_reg(f, window) > 5 * max(nearby_fft)


def test_fdh_fwh_fwd(wave_objs):
    (f1, obj1), (f2, obj2) = wave_objs
    assert (
        obj1.fdh() / obj1._freq_based_reg(.9 * f1, 60) > 5 and
        obj1.fdh() / obj1._freq_based_reg(1.1 * f1, 60) > 5)
    assert pytest.approx(obj1.fwh(), 0.1) == 0  # don't have week frequency
    assert pytest.approx(obj1.fwd(), 0.1) == 0

    assert (
        obj2.fwh() / obj2._freq_based_reg(.9 * f2, 60) > 5 and
        obj2.fwh() / obj2._freq_based_reg(1.1 * f2, 60) > 5)
    assert (
        obj2.fwd() / obj2._freq_based_reg(.9 * f2, 60 * 24) > 5 and
        obj2.fwd() / obj2._freq_based_reg(1.1 * f2, 60 * 24) > 5)


def test_get_regularity_indicators(null_obj, reg_obj, wave_objs):
    (f1, obj1), (f2, obj2) = wave_objs

    l1 = null_obj.get_regularity_indicators()
    l2 = reg_obj.get_regularity_indicators()
    l3 = obj1.get_regularity_indicators()

    assert l1 == []
    assert l2 == []
    assert sorted(l3) == sorted(
        ['pdh', 'fdh', 'fwh'] + ['pwd', 'ws1', 'ws2', 'ws3', 'fwd'])


def test_get_regularity(wave_objs):
    (f1, obj1), (f2, obj2) = wave_objs
    res = obj1.get_regularity()

    assert res['ws1'] == res['ws2'] == res['ws3'] == 1
    assert (
        pytest.approx(res['fwh']) ==
        pytest.approx(res['fwd']) ==
        pytest.approx(res['pwd']) == 0)


def test_jsd():
    obj1 = regularity.ActivityRegularity(
        ['red', 'blue', 'yellow'],
        [[1. / 2, 1. / 2, 0], [1. / 2, 0, 1. / 2]])

    obj2 = regularity.ActivityRegularity(
        ['red', 'blue', 'yellow'],
        [[.5, .5, 0], [0, 0.5, .5], [.5, .25, .25]])

    obj3 = regularity.ActivityRegularity(
        ['red', 'blue', 'yellow'],
        [[1.5, 1.5, 0], [0, 0.5, .5]])

    assert obj1.jsd() == 0.5
    assert pytest.approx(obj2.jsd()) == 0.38791850267113315
    assert pytest.approx(obj3.jsd(True)) == 0.40563906222956647


def test_concentration():
    obj1 = regularity.ActivityRegularity(
        ['forum', 'content'],
        [[1, 1], [1, 1]])
    obj2 = regularity.ActivityRegularity(
        ['forum', 'content'],
        [[3, 1], [3, 2]])

    assert obj1.concentration() == 0
    assert obj2.concentration() > obj1.concentration()