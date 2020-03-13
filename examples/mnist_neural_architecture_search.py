#!/usr/bin/env python
import uuid
import typing
from pathlib import Path

import tensorflow as tf
import tensorflow.keras as keras
#from tensorflow.examples.tutorials.mnist import input_data
from tfdbonas import Searcher, Trial

from tnng import Generator, MultiHeadLinkedListLayer
import tfcg


def create_model(network):
    model = keras.Sequential()
    for layer in network:
        model.add(layer[0])
    return model


def objectve(trial: Trial):
    network, (_, _) = trial.graph
    model = create_model(network)

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    model.fit(x_train, y_train, epochs=1, batch_size=128)
    out = model.evaluate(x_test,  y_test, verbose=2)
    accuracy = out[1]
    return accuracy

if __name__ == '__main__':
    m = MultiHeadLinkedListLayer()
    # graph created
    m.append_lazy(keras.layers.Flatten, [dict(input_shape=(28, 28)),])
    m.append_lazy(keras.layers.Dense, [dict(units=16),
                                       dict(units=32),
                                       dict(units=64),
                                       dict(units=128),
                                       dict(units=512),
                                       dict(units=1028),
    ])
    m.append_lazy(keras.layers.ReLU, [dict(),])
    m.append_lazy(keras.layers.Dense, [dict(units=16),
                                       dict(units=32),
                                       dict(units=64),
                                       dict(units=128),
                                       dict(units=512),
                                       dict(units=1028),

    ])
    m.append_lazy(keras.layers.ReLU, [dict(),])
    m.append_lazy(keras.layers.Dense, [dict(units=16),
                                       dict(units=32),
                                       dict(units=64),
                                       dict(units=128),
                                       dict(units=512),
                                       dict(units=1028),

    ])
    m.append_lazy(keras.layers.ReLU, [dict(),])
    m.append_lazy(keras.layers.Dense, [dict(units=16),
                                       dict(units=32),
                                       dict(units=64),
                                       dict(units=128),
                                       dict(units=512),
                                       dict(units=1028),
    ])
    m.append_lazy(keras.layers.ReLU, [dict(),])
    m.append_lazy(keras.layers.Dense, [dict(units=10, activation='softmax'),])
    g = Generator(m, dump_nn_graph=True)
    num_nodes = 10
    num_layer_type = 3
    searcher = Searcher()
    searcher.register_trial('graph', g)
    n_trials = 30
    model_kwargs = dict(
        num_nodes=num_nodes,
        input_channels=num_layer_type,
        n_train_epochs=400,
    )
    _ = searcher.search(objectve,
                        n_trials=n_trials,
                        deep_surrogate_model=f'tfdbonas.deep_surrogate_models:GCNSurrogateModel',
                        n_random_trials=10,
                        model_kwargs=model_kwargs)
    print(searcher.result)
    print('best_trial', searcher.best_trial)
    print('best_value', searcher.best_value)
