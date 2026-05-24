import math

from utils_plan_chunks.plan_chunks import plan_chunks


def test_default():
    adj = lambda length: length
    assert plan_chunks(adj, 20, 0, 99) == (20, 99, [20, 20, 20, 20, 19])
    assert plan_chunks(adj, 20, 0, 100) == (20, 100, [20, 20, 20, 20, 20])
    assert plan_chunks(adj, 20, 0, 101) == (20, 101, [20, 20, 20, 20, 20, 1])


def test_wan():
    adj = lambda length: (math.ceil((length - 1) / 4) * 4) + 1
    assert plan_chunks(adj, 81, 4, 157) == (81, 158, [81, 81])
    assert plan_chunks(adj, 81, 4, 158) == (81, 158, [81, 81])
    assert plan_chunks(adj, 81, 4, 159) == (81, 159, [81, 81, 5])


def test_ltx2():
    adj = lambda length: (math.ceil((length - 1) / 8) * 8) + 1
    assert plan_chunks(adj, 121, 8, 233) == (121, 234, [121, 121])
    assert plan_chunks(adj, 121, 8, 234) == (121, 234, [121, 121])
    assert plan_chunks(adj, 121, 8, 235) == (121, 235, [121, 121, 9])
