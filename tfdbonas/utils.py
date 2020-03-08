from enum import Flag, auto

from .trial import Trial


class Result:
    def __init__(self, n_remenbers: int = 10):
        self.trials = []
        self.score = []
        self.min_score: float = 0.0

    def push(self, score: float, trial: Trial):
        pass

    def top(self) -> float:
        return 0


class State(Flag):
    Initialized = auto()
    NotInitialized = auto()
