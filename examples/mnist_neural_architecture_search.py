#!/usr/bin/env python
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.examples.tutorials.mnist import input_data
from tfdbonas import Searcher, Trial

from tnng import Generator, MultiHeadLinkedListLayer
import tfcg


from kgcn.layers import GraphConv, GraphGather


class GCNSurrogateModel:
    def __init__(self, output_channles: int, hidden_channels: int=64):
        super(GCNSurrogateModel, self).__init__()
        self.gcn1 = GraphConv(16, adj_channel_num=1)
        self.gcn2 = GraphConv(32, adj_channel_num=1)
        self.gcn3 = GraphConv(64, adj_channel_num=1)
        self.gather = GraphGather()
        self.l1 = keras.layers.Dense(hidden_channels)
        self.l2 = keras.layers.Dense(output_channles)
        self.bases = None

    def last_layer(self, inputs, adj):
        x = self.gcn1(inputs, adj)
        x = tf.nn.sigmoid(x)
        x = self.gcn1(inputs, adj)
        x = tf.nn.sigmoid(x)
        x = self.gcn1(inputs, adj)
        x = tf.nn.sigmoid(x)
        x = self.gather(x)
        x = self.l1(x)
        self.bases = x
        return x

    def __call__(self, inputs, adj):
        x = self.last_layer(inputs, adj)
        x = self.l2(x)
        return x

    def _build_graph(self, xdim: int, ydim: int):
        if True:
            self.y_plh_train = tf.placeholder(tf.float32, shape=[None, ydim], name='ytrain')
            self.x_plh_train = tf.placeholder(tf.float32, shape=[None, xdim], name='xtrain')
            out = self(self.x_plh_train)
            mse_loss = tf.reduce_mean(tf.square(self.y_plh_train - out))
            self.train_loss = tf.train.AdamOptimizer(learning_rate=0.001).minimize(mse_loss)

    def train(self, xtrain=typing.List[Trial], ytrain=typing.List[float], n_epochs: int=None):
        if not n_epochs is None:
            n_epochs = self.n_train_epochs
        if True:
            bases = []
            with tf.Session(config=self.tf_config) as sess:
                sess.run(tf.global_variables_initializer())
                for _ in range(n_epochs):
                    for x, y in zip(xtrain, ytrain):
                        x = x.to_numpy().reshape(1, self.input_dim)
                        y = np.array(y, dtype=np.float32).reshape(1, 1)
                        sess.run(self.train_loss, feed_dict={self.x_plh_train: x, self.y_plh_train: y})
                for x, y in zip(xtrain, ytrain):
                    x = x.to_numpy().reshape(1, self.input_dim)
                    y = np.array(y, dtype=np.float32).reshape(1, 1)
                    bases.append(sess.run(self.bases, feed_dict={self.x_plh_train: x, self.y_plh_train: y}))
                bases = np.concatenate(bases)
                self.saver.save(sess, self.save_path)
        return bases

    def predict(self, xeval=typing.List[Trial]):
        bases = []
        with tf.Session(config=self.tf_config) as sess:
            self.saver.restore(sess, self.save_path)
            for x in xeval:
                x = x.to_numpy().reshape(1, self.input_dim)
                y = np.ones((1, 1), dtype=np.float32) # dummy input
                bases.append(sess.run(self.bases, feed_dict={self.x_plh_train: x, self.y_plh_train: y}))
            bases = np.concatenate(bases)
        return bases



def create_model(graph):
    print(graph)
    model = keras.Sequential()
    for layer in graph:
        model.add(layer[0])
    return model

def objectve(trial: Trial):
    model = create_model(trial.graph)

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print(model)
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
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
                                       dict(units=128)])
    m.append_lazy(keras.layers.ReLU, [dict(),])
    m.append_lazy(keras.layers.Dense, [dict(units=16),
                                       dict(units=32),
                                       dict(units=64),
                                       dict(units=128)])
    m.append_lazy(keras.layers.ReLU, [dict(),])
    m.append_lazy(keras.layers.Dense, [dict(units=16),
                                       dict(units=32),
                                       dict(units=64),
                                       dict(units=128)])
    m.append_lazy(keras.layers.ReLU, [dict(),])
    m.append_lazy(keras.layers.Dense, [dict(units=16),
                                       dict(units=32),
                                       dict(units=64),
                                       dict(units=128)])
    m.append_lazy(keras.layers.ReLU, [dict(),])
    m.append_lazy(keras.layers.Dense, [dict(units=10),])
    g = Generator(m)
    print(g[0])
    searcher = Searcher()
    searcher.register_trial('graph', g)
    n_trials = 30
    model_kwargs = dict(
        input_dim=10,
        n_train_epochs=200,
    )
    _ = searcher.search(objectve,
                        n_trials=n_trials,
                        deep_surrogate_model='tfdbonas.deep_surrogate_models:SimpleNetwork',
                        n_random_trials=10,
                        model_kwargs=model_kwargs)
    print(searcher.result)
    print('best_trial', searcher.best_trial)
    print('best_value', searcher.best_value)
