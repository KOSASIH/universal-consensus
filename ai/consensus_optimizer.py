# ai/consensus_optimizer.py
import tensorflow as tf
from tensorflow import keras

class ConsensusOptimizer(keras.Model):
    def __init__(self):
        super(ConsensusOptimizer, self).__init__()
        self.dense1 = keras.layers.Dense(64, activation='relu')
        self.dense2 = keras.layers.Dense(64, activation='relu')
        self.output_layer = keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)
