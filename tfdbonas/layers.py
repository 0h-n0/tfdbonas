import tensorflow as tf


class GCNConv(tf.keras.layers.Layer):
    def __init__(self, output_channels: int):
        super(GCNConv, self).__init__()
        self.output_channels = output_channels

    def build(self, input_shape):
        self.w = self.add_weight(name='kernel',
                                 shape=(int(input_shape[2]), int(self.output_channels)),
                                 initializer=self.initializer,
                                 trainable=True)
        self.bias = self.add_weight(name='bias',
                                 shape=(int(input_shape[2]), int(self.output_channels)),
                                 initializer=self.initializer,
                                 trainable=True)
        pass


    def call(self, input):
        pass
