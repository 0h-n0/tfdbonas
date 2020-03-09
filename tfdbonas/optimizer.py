import math
import typing
import random
from enum import Flag, auto

from .trial import Trial, TrialGenerator
from .utils import State, load_class

import numpy as np
import scipy.optimize
import scipy.stats


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
        deep_surrogate_model = kwargs['deep_surrogate_model']
        _ = self._random_search(objective, n_random_trials)
        results = self._bayes_search(objective, n_trials - n_random_trials, deep_surrogate_model)
        return self.results

    def _random_search(self,
                       objective: typing.Callable[[Trial], float],
                       n_trials: int) -> typing.List[int]:
        assert len(self._trials_indices) >= n_trials, (f'len(self._trials_indices) >= n_trials:'
                                                       f' {len(self._trials_indices)} >= {n_trials}')
        trial_indices = random.sample(self._trials_indices, n_trials)
        for i in trial_indices:
            self.results[i] = objective(self.trial_generator[i])
            self._trials_indices.remove(i)
        self._searched_trial_indices += trial_indices
        self._state = State.Initialized
        return self._trials_indices # return remained trials

    def _bayes_search(self,
                      objective: typing.Callable[[Trial], float],
                      n_trials: int,
                      deep_surrogate_model_path: str) -> typing.List[int]:
        deep_surrogate_model = load_class(deep_surrogate_model_path)()
        assert self._state == State.Initialized, ('not initialied: please call '
                                                  'self.random_search() before calling bayes_search.')
        assert len(self._searched_trial_indices) != 0, 'Before searching, you have to run random search.'
        for _ in range(n_trials):
            trained_bases = self._train_deep_surrogate_model(
                self._searched_trial_indices,
                self.results,
                deep_surrogate_model)
            n_samples = len(self._searched_trial_indices)
            n_features = self.trial_generator.n_features
            params = self._update_mll_params(trained_bases,
                                             self._searched_trial_indices,
                                             self.results,
                                             n_samples,
                                             n_features)
            mean, var = self._predict(params, self._trials_indices)
            acq_values = self._calc_acq_value(mean, var)
            next_sample_index = np.argmax(acq_values)
            self._searched_trial_indices.append(next_sample_index)
            self._trials_indices.remove(next_sample_index)
            self.results[next_sample_index] = objective(self.trial_generator[next_sample_index])
        return self.results

    def _train_deep_surrogate_model(self,
                                    searched_trial_indices: typing.List[int],
                                    results: typing.Dict[int, float],
                                    deep_surrogate_model):
        assert len(searched_trial_indices) == len(results), ('invalid inputs, searched_trial_indices[{searched_trial_indices}] '
                                                             'and results[{results}] must be the same length.')
        searched_trials = [self.trial_generator[i] for i in searched_trial_indices]
        trained_bases = deep_surrogate_model.train(searched_trials, results)
        return trained_bases

    def _predict(self, params, remained_trial_indicees):
        mean, var = 0, 0
        return mean, var

    def _calc_acq_value(self, mean, var):
        min_val = np.min(mean)
        return min_val# self.acq_func(mean, var, min_val)

    # def predict(self, x):
    #     _x = copy.deepcopy(x)
    #     _, beta = torch.exp(self.params).float()
    #     _x = (_x - self.mean_x) / self.std_x
    #     phi = self.nn.partial_forward(_x)
    #     mean = torch.matmul(phi, self.m)
    #     var = torch.diag(torch.matmul(torch.matmul(phi, self.K_inv), phi.t()) + 1 / beta)
    #     mean = mean * self.std_y + self.mean_y
    #     var = var * self.std_y ** 2
    #     #var = var.reshape(mean.shape[0], mean.shape[1])
    #     return mean.detach(), var.detach()

    def _update_mll_params(self, bases, searched_trial_indices,
                           results, n_samples, n_features):

        y_values = [results[i] for i in searched_trial_indices]
        params = scipy.optimize.fmin(self._calc_marginal_log_likelihood,
                                     np.random.rand(2),
                                     args=(bases, y_values, n_samples, n_features))
        return params

    def _calc_marginal_log_likelihood(self,
                                      theta,
                                      phi,
                                      y_values,
                                      n_samples,
                                      n_features):
        # TODO: input type check
        assert theta.size == 2, f"invalid input: theta => {theta}"
        assert len(theta.shape) == 1, f"invalid input: theta => {theta}"
        assert y_values.size == n_samples, f"invalid input: y_values.size => {y_values.size}"
        alpha, beta = np.exp(theta)

        # calculate K matrix
        identity = np.eye(phi.shape[1])
        phi_t = phi.transpose(1, 0)
        k_mat = beta * np.matmul(phi_t, phi) + alpha * identity

        # calculate mat
        k_inv = np.linalg.inv(k_mat)
        mat = beta * np.matmul(k_inv, phi_t)
        mat = np.matmul(mat, y_values)

        self.mat = mat
        self.k_inv = k_inv
        print(mat.shape)
        mll = n_features / 2. * np.log(alpha)
        mll += n_samples / 2. * np.log(beta)
        mll -= n_samples / 2. * np.log(2 * math.pi)
        mll -= beta / 2. * np.linalg.norm(y_values - np.matmul(phi, mat))
        mll -= alpha / 2. * mat.dot(mat)
        mll -= 0.5 * np.log(np.linalg.det(k_mat))
        return -mll
