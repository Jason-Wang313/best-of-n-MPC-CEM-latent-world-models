from boncem.metrics import regret, selected_tail_gap


def test_regret_formula():
    assert regret(3.5, 1.25) == 2.25


def test_selected_tail_gap_formula():
    assert selected_tail_gap(4.0, -1.5) == 5.5
