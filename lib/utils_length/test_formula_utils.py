import math

from formula_utils import plan_chunks


def test_default():
    adj = lambda length: length
    assert plan_chunks(adj, 20, 0, 99) == ([20, 20, 20, 20, 19], 99, 20)
    assert plan_chunks(adj, 20, 0, 100) == ([20, 20, 20, 20, 20], 100, 20)
    assert plan_chunks(adj, 20, 0, 101) == ([20, 20, 20, 20, 20, 1], 101, 20)


def test_wan():
    adj = lambda length: (math.ceil((length - 1) / 4) * 4) + 1
    assert plan_chunks(adj, 81, 4, 157) == ([81, 81], 158, 81)
    assert plan_chunks(adj, 81, 4, 158) == ([81, 81], 158, 81)
    assert plan_chunks(adj, 81, 4, 159) == ([81, 81, 5], 159, 81)


def test_ltx2():
    adj = lambda length: (math.ceil((length - 1) / 8) * 8) + 1
    assert plan_chunks(adj, 121, 8, 233) == ([121, 121], 234, 121)
    assert plan_chunks(adj, 121, 8, 234) == ([121, 121], 234, 121)
    assert plan_chunks(adj, 121, 8, 235) == ([121, 121, 9], 235, 121)
