#!/usr/bin/env python
import uuid
import typing
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
#from tensorflow.examples.tutorials.mnist import input_data
from tfdbonas import Searcher, Trial

from tnng import Generator, MultiHeadLinkedListLayer
import tfcg


def create_model(network, inputs):
    xx = inputs
    for layer in network:
        if len(layer) == 4:
            if layer[2] is None and layer[3] is None:
                x1 = layer[0](xx[0])
                x2 = layer[1](xx[1])
                xx = [x1, x2, xx[2], xx[3]]
            elif layer[3] is None:
                x1 = layer[0](xx[0])
                x2 = layer[1](xx[1])
                x3 = layer[2](xx[2])
                xx = [x1, x2, x3, xx[3]]
            else:
                x1 = layer[0](xx[0])
                x2 = layer[1](xx[1])
                x3 = layer[2](xx[2])
                x4 = layer[3](xx[3])
                xx = [x1, x2, x3, x4]
        elif len(layer) == 3:
            x1 = keras.layers.concatenate([xx[0], xx[1]], axis=1)
            xx = [x1, xx[2], xx[3]]
        elif len(layer) == 2:
            x1 = keras.layers.concatenate([xx[0], xx[1]], axis=1)
            xx = [x1, xx[2]]
        elif len(layer) == 1:
            if layer[0] == 'concat':
                xx = keras.layers.concatenate(xx, axis=1)
            else:
                xx = layer[0](xx)
    return xx


def objectve(trial: Trial):
    network, (_, _) = trial.graph
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_trains = np.split(x_train, 4, axis=1)
    x_tests = np.split(x_test, 4, axis=1)
    inputs = [keras.layers.Input(shape=(7, 28), name=f'input{i}')
              for i in range(1, 5)]
    out = create_model(network, inputs)
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    model = keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_trains, y_train, epochs=1, batch_size=128)
    out = model.evaluate(x_tests, y_test, verbose=2)
    accuracy = out[1]
    return accuracy


def create_modal():
    m = MultiHeadLinkedListLayer()
    # graph created
    dense_kwargs = [dict(units=32), dict(units=128), dict(units=512)]
    m.append_lazy(keras.layers.Flatten, [dict(input_shape=(14, 28)),])
    m.append_lazy(keras.layers.Dense, dense_kwargs)
    m.append_lazy(keras.layers.ReLU, [dict(),])
    m.append_lazy(keras.layers.Dense, dense_kwargs)
    m.append_lazy(keras.layers.ReLU, [dict(),])
    return m

if __name__ == '__main__':
    m1 = create_modal()
    m2 = create_modal()
    m3 = create_modal()
    m4 = create_modal()
    m = m1 + m2 + m3 + m4
    m.append_lazy(keras.layers.Dense, [dict(units=10, activation='softmax'),])
    g = Generator(m, dump_nn_graph=True)
    g.draw_graph('/home/ono/Dropbox/test.png')
    num_nodes = 24
    num_layer_type = 4
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
