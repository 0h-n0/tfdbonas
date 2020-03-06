import tensorflow as tf
import tensorflow.keras.layers as L

def get_stellargraph_gcn_class():
    # for tensorflow 2.0
    from stellargraph.layer import GraphConvolution

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
    from kgcn import GraphConv, GraphGather

    class GCNSurrogateModel(tf.keras.models.Model):
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
    return GCNSurrogateModel

if __name__ == "__main__":
    g = GCNSurrogateModel()
    features = tf.placeholder(tf.float32, shape=(None, 25, 10))
    adj = tf.placeholder(tf.float32, shape=(1, 25, 25))
    o = g([features, [0,], adj])
    print(o)
