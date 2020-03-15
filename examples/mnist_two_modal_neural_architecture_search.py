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
        if len(layer) == 2:
            x1 = layer[0](xx[0])
            x2 = layer[1](xx[1])
            xx = [x1, x2]
        elif len(layer) == 1:
            if layer[0] == 'concat':
                xx = keras.layers.concatenate(xx, axis=1)
                print(xx.shape)
            else:
                xx = layer[0](xx)
    return xx


def objectve(trial: Trial):
    network, (_, _) = trial.graph
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train_1, x_train_2 = np.split(x_train, 2, axis=1)
    x_test_1, x_test_2 = np.split(x_test, 2, axis=1)
    left_input = keras.layers.Input(shape=(14, 28), name='left_input')
    right_input = keras.layers.Input(shape=(14, 28), name='right_input')
    out = create_model(network, [left_input, right_input])
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    model = keras.Model(inputs=[left_input, right_input], outputs=out)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit([x_train_1, x_train_2], y_train, epochs=1, batch_size=128)
    out = model.evaluate([x_test_1, x_test_2], y_test, verbose=2)
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
    m = m1 + m2
    m.append_lazy(keras.layers.Dense, [dict(units=10, activation='softmax'),])
    g = Generator(m, dump_nn_graph=True)
    g.draw_graph('/home/ono/Dropbox/test.png')
    num_nodes = 12
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
