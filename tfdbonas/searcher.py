import pathlib
import random
import typing
from enum import Flag, auto

from .trial import (Trial,
                    TrialGenerator)


class SearcherState(Flag):
    Initialized = auto()
    NotInitialized = auto()


class Searcher:
    def __init__(self):
        self.trial_generator = TrialGenerator()
        self.search_algorithm = 'DNGO'
        self._trials_indices: typing.List[int] = []
        self._searched_trial_indices: typing.List[int] = []
        self.results: typing.Dict[int, float] = {}
        self._state = SearcherState.NotInitialized

    def register_trial(self, name: str, trial: list):
        self.trial_generator.register(name, trial)

    def initialize_trials(self) -> typing.List[int]:
        self._trials_indices = list(range(len(self.trial_generator)))
        self._searched_trial_indices: typing.List[int] = []
        self._state = SearcherState.Initialized
        return self._trials_indices

    def random_search(self, objective: typing.Callable[[Trial], float], n_trials: int) -> typing.List[int]:
        assert self._state == SearcherState.Initialized, 'not initialied: please self.initialize_trials() before searching.'
        assert len(self._trials_indices) >= n_trials, f'len(self._trials_indices) >= n_trials: {len(self._trials_indices)} >= {n_trials}'
        trial_indices = random.sample(self._trials_indices, n_trials)
        for i in trial_indices:
            self.results[i] = objective(self.trial_generator[i])
            self._trials_indices.remove(i)
        self._searched_trial_indices += trial_indices
        return self._trials_indices # return remained trials

    def search(self, objective: typing.Callable[[Trial], float], n_trials: int):
        assert self._state == SearcherState.Initialized, 'not initialied: please self.initialize_trials() before searching.'
        assert len(self._searched_trial_indices) != 0, 'Before do this, you have to run random search'
        pass

    def __len__(self):
        return len(self.trial_generator)

    @staticmethod
    def read_config(path: pathlib.Path or str):
        pass

    @staticmethod
    def _read_toml(path: pathlib.Path):
        pass

    @staticmethod
    def _read_yaml(path: pathlib.Path):
        pass

    def dump(self, path: pathlib.Path or str):
        pass

    def _dump_yaml(self, path: pathlib.Path):
        pass

    def _dump_toml(self, path: pathlib.Path):
        pass
