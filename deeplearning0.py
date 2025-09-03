import tensorflow as tf
from tensorflow import keras


class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__()

        self.w = self.add_weight([input_dim, output_dim])
        self.b = self.add_weight([1, output_dim])
    
    def call(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b

        output = tf.math.sigmoid(z)
        return output
        
    
    