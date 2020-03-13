#!/usr/bin/env python
# ref: https://github.com/optuna/optuna/blob/master/examples/quadratic_simple.py
import optuna
import numpy as np

def objective(trial):
    x = trial.suggest_categorical('x', np.arange(-10, 10, 0.1))
    y = trial.suggest_categorical('y', np.arange(-10, 10, 0.1))
    return x**2 + y**2


if __name__ == '__main__':
    # Let us minimize the objective function above.
    print('Running 10 trials...')
    study = optuna.create_study()
    study.optimize(objective, n_trials=10)
    print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))

    # We can continue the optimization as follows.
    print('Running 20 additional trials...')
    study.optimize(objective, n_trials=20)
    print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))

    # We can specify the timeout instead of a number of trials.
    print('Running additional trials in 2 seconds...')
    study.optimize(objective, timeout=2.0)
    print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))
