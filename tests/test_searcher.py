from tfdbonas.searcher import Searcher


def test_searcher():
    def ovjective(trial):
        return trial.lr * trial.bs

    params = [
            ['lr', [0.1 * i for i in range(10)]],
            ['bs', [64 * i for i in range(10)]],
    ]

    s = Searcher()

    for p in params:
        s.register_trial(p[0], p[1])
    assert len(s) == 100
    assert s.initialize_trials() == list(range(100))
    assert len(s.random_search(ovjective, 10)) == 90
    assert len(s._searched_trial_indices) == 10

def test_searcher_with_edge_case():
    def ovjective(trial):
        return trial.lr

    params = [
            ['lr', [0.1]]
    ]

    s = Searcher()

    for p in params:
        s.register_trial(p[0], p[1])
    assert len(s) == 1
    assert s.initialize_trials() == list(range(1))
    assert len(s.random_search(ovjective, 1)) == 0
    s.random_search(ovjective, 2)
