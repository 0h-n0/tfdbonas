from pathlib import Path

from tfdbonas.utils import load_class


class HogeClass:
    def __init__(self):
        pass


def test_load_class():
    path = 'tfdbonas.deep_surrogate_models:SimpleNetwork'
    c = load_class(path)
    c()
