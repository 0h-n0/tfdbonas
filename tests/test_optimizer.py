import pytest

from tfdbonas.optimizer import DNGO
from tfdbonas.trial import TrialGenerator


@pytest.fixture(
    params=[
        {'n_trials': 10, 'n_remains': 990},
        {'n_trials': 500, 'n_remains': 500},
        {'n_trials': 1, 'n_remains': 999},
        {'n_trials': 1000, 'n_remains': 0},
    ]
)
def correct_tirals_test_case(request):
    return request.param

def test_dngo_random_search(correct_tirals_test_case):
    data = correct_tirals_test_case
    def objective(trial):
        return trial.lr * trial.bs

    params = [
        ['lr', [0.1 * i for i in range(10)]],
        ['bs', [64 * i for i in range(10)]],
        ['network', [64 * i for i in range(10)]],
    ]
    t = TrialGenerator()
    for inputs in params:
        t.register(inputs[0], inputs[1])
    algo = DNGO(t)
    assert len(t) == 1000
    assert data['n_remains'] == len(algo._random_search(objective, data['n_trials']))
    assert data['n_trials'] == len(algo._searched_trial_indices)
    assert data['n_trials'] == len(algo.results)

def test_dngo_bayes_search():
    def objective(trial):
        return trial.lr * trial.bs * trial.network
    params = [
        ['lr', [0.1 * i for i in range(10)]],
        ['bs', [64 * i for i in range(10)]],
        ['network', [64 * i for i in range(10)]],
    ]
    t = TrialGenerator()
    for inputs in params:
        t.register(inputs[0], inputs[1])
    algo = DNGO(t)
    n_random = 10
    n_bayes = 10
    algo._random_search(objective, n_random)
    algo._bayes_search(objective, n_bayes)
    pass
