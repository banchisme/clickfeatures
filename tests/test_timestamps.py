from clickfeatures import timestamps
import pytest
import random
import numpy as np


@pytest.fixture
def ts_obj():
    ts = [0, 60, 120, 240, 300]
    ws = [1, 2, 100, 1, 1]
    return timestamps.TimeStamps(ts, ws)


@pytest.fixture
def null_obj():
    return timestamps.TimeStamps([], end=1)


@pytest.fixture
def rand_obj():
    length = 60 * 24  # 1 day
    ts = random.sample(range(length), length / 2)
    ws = random.sample(range(length), length / 2)
    obj = timestamps.TimeStamps(ts, ws, end=length)
    return obj


def test_quantify(null_obj):
    assert null_obj.quantify('hour') == 60
    assert null_obj.quantify('day') == 60 * 24
    assert null_obj.quantify('week') == 60 * 24 * 7
    with pytest.raises(ValueError):
        null_obj.quantify('second')


def test_active(rand_obj):
    r"""
        this tests given a random time series array (reg_obj) that is half-activated,
        if we check a number of random set timestamps, if the _active method return
        half of the random timestamps as active
    """
    size = 20000
    xs = np.random.rand(size) * rand_obj.get_duration()
    res = [rand_obj.active(x, 1) for x in xs]

    assert 0.5 == pytest.approx(1. * sum(res) / size, 0.05)


def test_group():
    ts = [i * 30 for i in range(48)]
    ws = range(48)
    obj = timestamps.TimeStamps(ts, ws)

    expected_ts = [[30 * i, 30 * (i + 1)] for i in range(0, 48, 2)]
    expected_ws = [[i, i + 1] for i in range(0, 48, 2)]

    result_ts, result_ws = obj.group(60)
    assert expected_ts == result_ts
    assert expected_ws == result_ws


def test_agg():
    ts = range(24 * 60)
    ws = range(24 * 60)
    obj = timestamps.TimeStamps(ts, ws)

    expects = [sum(range(60)) + 60 * 60 * i for i in range(24)]
    _, results = obj.aggregate(60, sum)
    assert expects == results


def test_to_signal():
    ts = [1, 5, 10]
    ws = [1, 1, 1]
    obj = timestamps.TimeStamps(ts, ws)

    expect_ts = range(11)
    expect_ws = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    result_ts, result_ws = obj.aggregate(1, sum)
    assert expect_ts == result_ts
    assert expect_ws == result_ws
