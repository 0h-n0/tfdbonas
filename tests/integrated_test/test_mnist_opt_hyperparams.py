#!/usr/bin/env python
import warnings

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.examples.tutorials.mnist import input_data
from tfdbonas import Searcher, Trial


def create_model(hidden_size=128):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(hidden_size, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model


def objectve(trial: Trial):
    model = create_model(trial.hidden_size)
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    opt = tf.keras.optimizers.SGD(learning_rate=trial.lr)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    model.fit(x_train, y_train, epochs=1, batch_size=trial.batchsize)
    out = model.evaluate(x_test,  y_test, verbose=2)
    accuracy = out[1]
    return accuracy


def test_main():
    searcher = Searcher()
    searcher.register_trial('hidden_size', [64, 128, 256, 512, 1024])
    searcher.register_trial('batchsize', [32, 64, 128, 256, 512, 1024])
    searcher.register_trial('lr', [0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    n_trials = 30

    model_kwargs = dict(
        input_dim=3,
        n_train_epochs=200,
    )
    _ = searcher.search(objectve,
                        n_trials=n_trials,
                        deep_surrogate_model='tfdbonas.deep_surrogate_models:SimpleNetwork',
                        n_random_trials=10,

                             model_kwargs=model_kwargs)
    assert len(searcher.result) == n_trials
    warnings.warn('results = {}'.format(searcher.result))
    warnings.warn('best_trial {}'.format(searcher.best_trial))
    warnings.warn('best_value {}'.format(searcher.best_value))
