import uuid
import typing

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L

from .trial import Trial

class BaseDeepSurrogateModel(tf.keras.models.Model):
    def __init__(self):
        super(BaseDeepSurrogateModel, self).__init__()

    def train(self, x, y):
        pass



def get_stellargraph_gcn_class():
    # for tensorflow 2.0
    from stellargraph.layer import GraphConvolution

    class GCNSurrogateModel(BaseDeepSurrogateModel):
        def __init__(self, output_channles: int, hidden_channels: int=64):
            super(GCNSurrogateModel, self).__init__()
            self.gcn1 = GraphConvolution(16, 'tanh')
            self.gcn2 = GraphConvolution(32, 'tanh')
            self.gcn3 = GraphConvolution(64, 'tanh', final_layer=True)
            self.l1 = L.Dense(hidden_channels)
            self.l2 = L.Dense(output_channles)

        def last_layer(self, inputs):
            """
            See Definition of GraphConvolution
            - https://stellargraph.readthedocs.io/en/stable/_modules/stellargraph/layer/gcn.html#GraphConvolution.call
            """
            features, out_indices, *As = inputs
            x = self.gcn1(inputs)
            x = self.gcn2([x, out_indices, As])
            x = self.gcn3([x, out_indices, As])
            self.bases = x
            return x
            return self.l1(x)

        def call(self, inputs):
            x = self.last_layer(inputs)
            x = self.l2(x)
            return x
    return GCNSurrogateModel


def get_kgcn_gcn_class():
    # for tensorflow 1.x
    from kgcn.layers import GraphConv, GraphGather

    class GCNSurrogateModel(BaseDeepSurrogateModel):
        def __init__(self, output_channles: int, hidden_channels: int=64):
            super(GCNSurrogateModel, self).__init__()
            self.gcn1 = GraphConv(16, adj_channel_num=1)
            self.gcn2 = GraphConv(32, adj_channel_num=1)
            self.gcn3 = GraphConv(64, adj_channel_num=1)
            self.gather = GraphGather()
            self.l1 = L.Dense(hidden_channels)
            self.l2 = L.Dense(output_channles)
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

        def train(self, x, y):
            pass

    return GCNSurrogateModel


class SimpleNetwork:
    def __init__(self,
                 input_dim: int = 4,
                 output_dim: int = 1,
                 hidden_dim: int = 32,
                 activation='tanh',
                 n_train_epochs: int = 100,
                 save_path=f'/tmp/simplenetwork-{uuid.uuid1()}.ckpt'):
        self.first_layer = tf.keras.models.Sequential([
            L.Dense(32, activation),
            L.Dense(64, activation),
            L.Dense(hidden_dim, activation)])
        self.last_layer = L.Dense(output_dim)
        self.tf_config = tf.ConfigProto(log_device_placement=False,
                                        gpu_options=tf.GPUOptions(
                                            allow_growth=True,
                                        ))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_train_epochs = n_train_epochs
        self._build_graph(input_dim, output_dim)
        self.saver = tf.train.Saver()
        self.save_path = save_path

    def __call__(self, x):
        self.bases = self.first_layer(x)
        return self.last_layer(self.bases)

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


if __name__ == "__main__":
    gcn_class = get_kgcn_gcn_class()
    g = gcn_class(32)
    features = tf.placeholder(tf.float32, shape=(None, 25, 10))
    adj = tf.placeholder(tf.float32, shape=(1, 25, 25))
    o = g([features, [0,], adj])
    print(o)
