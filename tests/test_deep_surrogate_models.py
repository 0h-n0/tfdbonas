from tfdbonas.deep_surrogate_models import get_kgcn_gcn_class


def test_get_kgcn_gcn_class():
    import tensorflow as tf
    gcn_class = get_kgcn_gcn_class()
    g = gcn_class(32)
    features = tf.placeholder(tf.float32, shape=(1, 25, 10))
    adj = [tf.sparse_placeholder(tf.float32, shape=(25, 25)), ]
    o = g(features, [adj,])
    print(g.bases.shape)
    assert (1, 32) == o.shape
    assert (1, 64) == g.bases.shape
