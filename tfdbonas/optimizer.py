import typing
import random
from enum import Flag, auto

from .trial import Trial, TrialGenerator
from .utils import State

import numpy as np


class OptimizerType(Flag):
    DNGO = auto()


class DNGO:
    def __init__(self, trial_generator):
        self._trials_indices = list(range(len(trial_generator)))
        self.trial_generator = trial_generator
        self._state = State.NotInitialized
        self._searched_trial_indices: typing.List[int] = []
        self.results: typing.Dict[int, float] = {}

    def run(self, objective: typing.Callable[[Trial], float],
            n_trials: int, **kwargs):
        n_random_trials = kwargs['n_random_trials']
        self._random_search(objective, n_random_trials)
        self._bayes_search(objective, n_trials - n_random_trials)
        return 0

    def _random_search(self,
                      objective: typing.Callable[[Trial], float],
                      n_trials: int) -> typing.List[int]:
        assert len(self._trials_indices) >= n_trials, f'len(self._trials_indices) >= n_trials: {len(self._trials_indices)} >= {n_trials}'
        trial_indices = random.sample(self._trials_indices, n_trials)
        for i in trial_indices:
            self.results[i] = objective(self.trial_generator[i])
            self._trials_indices.remove(i)
        self._searched_trial_indices += trial_indices
        self._state = State.Initialized
        return self._trials_indices # return remained trials

    def _bayes_search(self,
                     objective: typing.Callable[[Trial], float],
                     n_trials: int) -> typing.List[int]:

        assert self._state == State.Initialized, 'not initialied: please self.initialize_trials() before searching.'
        assert len(self._searched_trial_indices) != 0, 'Before searching, you have to run random search.'
        for _ in range(n_trials):
            self._deep_bayes_train(self._searched_trial_indices,
                                   self.results)
            # mean, var = self.predict(self.to_tensor(self.trial.remains()))
            # acq = self.calc_acq_value(mean, var)
            # next_sample = np.argmax(acq)
            # t = self.trial.create_trainer(next_sample)
            # y = t.run()
            # print(f'acc = {y}')
            # self.train_y.append(-y) # change to minimum problem
            # self.train_x = self.trial.get_delete_list()

    def _deep_bayes_train(self,
                          searched_trial_indices,
                          results):
        for i in searched_trial_indices:
            print(results[i],
                  self.trial_generator[i])

    def calc_acq_value(self, mean, var):
        min_val = torch.min(self.norm_y)
        return self.acq_func(mean, var, min_val)

    def predict(self, x):
        _x = copy.deepcopy(x)
        _, beta = torch.exp(self.params).float()
        _x = (_x - self.mean_x) / self.std_x
        phi = self.nn.partial_forward(_x)
        mean = torch.matmul(phi, self.m)
        var = torch.diag(torch.matmul(torch.matmul(phi, self.K_inv), phi.t()) + 1 / beta)
        mean = mean * self.std_y + self.mean_y
        var = var * self.std_y ** 2
        #var = var.reshape(mean.shape[0], mean.shape[1])
        return mean.detach(), var.detach()

    def _calc_marginal_log_likelihood(self, theta ,phi=None, x=None, y=None):
        pass
