import tensorflow as tf
import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.input_size = None
        self.layers = None
        self.output_size = None
        self.layers_activation = None
        self.train_set = None
        self.train_labels = None
        self.test_set = None
        self.test_labels = None

        self.model = None

    def create_network(self, input_size, layers, output_size, layers_activation):
        self.input_size = input_size
        self.layers = layers
        self.output_size = output_size
        self.layers_activation = layers_activation

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(self.input_size,)))
        i = 0
        for layer in layers:
            self.model.add(tf.keras.layers.Dense(layer, activation=layers_activation[i]))
            i += 1
        self.model.add(tf.keras.layers.Dense(self.output_size, activation=layers_activation[-1]))

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train(self, train_set, lambdas, number_of_epochs):
        self.train_set = train_set
        self.train_labels = []

        train_inputs = []
        for example in train_set:
            self.train_labels.append(0.9999 if example.has_parkinson else 0)
            train_input = []
            for lam in lambdas:
                train_input += (lam(example))
            train_inputs.append(train_input)

        self.model.fit(np.array(train_inputs), np.array(self.train_labels), epochs=number_of_epochs)

    def test(self, test_set, lambdas):
        self.test_set = test_set
        self.test_labels = []

        test_inputs = []
        for example in test_set:
            self.test_labels.append(0.9999 if example.has_parkinson else 0)
            test_input = []
            for lam in lambdas:
                test_input.append(lam(example))
            test_inputs.append(test_input)

        test_loss, test_acc = self.model.evaluate(np.array(test_inputs), np.array(self.test_labels), verbose=2)

        print('\nTest accuracy:', test_acc)
