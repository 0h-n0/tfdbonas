import pathlib
import typing

from .trial import (Trial,
                    TrialGenerator)
from .optimizer import (OptimizerType,
                        DNGO)
from .utils import State



class Searcher:
    def __init__(self, search_algorithm=OptimizerType.DNGO):
        self.trial_generator = TrialGenerator()
        self.search_algorithm = search_algorithm
        self._state = State.NotInitialized

    def register_trial(self, name: str, trial: list):
        self.trial_generator.register(name, trial)

    def search(self,
               objective: typing.Callable[[Trial], float],
               n_trials: int, **kwargs):

        if OptimizerType.DNGO == self.search_algorithm:
            Optimizer = DNGO
            if not 'deep_surrogate_model' in kwargs.keys():
                raise ValueError("set 'deep_surrogate_model' in input kwargs'")
            if not 'n_random_trials' in kwargs.keys():
                raise ValueError("set 'n_random_trials' in input kwargs'")
        else:
            raise NotImplementedError("supported optimizer: DNGO")
        optimizer = Optimizer(self.trial_generator)
        result = optimizer.run(objective, n_trials, **kwargs)
        return result

    def __len__(self):
        return len(self.trial_generator)
