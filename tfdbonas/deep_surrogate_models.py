import tensorflow as tf
import tensorflow.keras.layers as L
from stellargraph.layer import GraphConvolution
from kgcn import GCN

class GCNSurrogateModel(tf.keras.models.Model):
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
        return self.l1(x)

    def call(self, inputs):
        x = self.last_layer(inputs)
        x = self.l2(x)
        return x

if __name__ == "__main__":
    g = GCNSurrogateModel()
    features = tf.placeholder(tf.float32, shape=(None, 25, 10))
    adj = tf.placeholder(tf.float32, shape=(1, 25, 25))
    o = g([features, [0,], adj])
    print(o)
