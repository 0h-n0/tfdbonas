import unittest

import pytest
import numpy as np

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


class TestDNGO(unittest.TestCase):
    @staticmethod
    def objective(trial) -> float:
        return trial.hidden1 * trial.hidden2 * trial.lr * trial.batchsize

    def setUp(self):
        params = [
            ['hidden1', [16, 32, 64, 128]],
            ['hidden2', [16, 32, 64, 128]],
            ['lr', [0.1 * i for i in range(10)]],
            ['batchsize', [64 * i for i in range(10)]],
        ]
        self.trial_generator = TrialGenerator()
        for inputs in params:
            self.trial_generator.register(inputs[0], inputs[1])

    def test_dngo_bayes_search(self):
        algo = DNGO(self.trial_generator)
        n_random = 10
        n_bayes = 10
        path = 'tfdbonas.deep_surrogate_models:SimpleNetwork'
        algo._random_search(TestDNGO.objective, n_random)
        #algo._bayes_search(TestDNGO.objective, n_bayes, path)

    def test__calc_marginal_log_likelihood(self):
        optimizer = DNGO(self.trial_generator)
        n_random = 10
        n_bayes = 10
        path = 'tfdbonas.deep_surrogate_models:SimpleNetwork'
        theta = np.random.rand(2)
        phi = np.random.rand(10, 10)
        y_values = np.random.rand(10)
        optimizer._calc_marginal_log_likelihood(theta, phi, y_values, 10, 10)

        #algo._random_search(TestDNGO.objective, n_random)
        #algo._bayes_search(TestDNGO.objective, n_bayes, path)
