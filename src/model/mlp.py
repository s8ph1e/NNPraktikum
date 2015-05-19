
import time

import numpy as np

from util.activation_functions import Activation
from model.layer import Layer
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier



class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 cost='crossentropy', learningRate=0.01, epochs=50):

        # Build up the network from specific layers
        # Here is an example
        # Should read those configuration from the command line or config file
        self.layers = []

        if layers is None:
            inputLayer = Layer(2, 2, weights=inputWeights)
            hiddenLayer = Layer(2, 2)
            if outputTask != 'classification':
                outputLayer = LogisticLayer(2, 1,
                                            activation=outputActivation,
                                            isClassifierLayer=False)
            else:
                outputLayer = LogisticLayer(2, 1,
                                            activation=outputActivation)

            self.layers.append(inputLayer)
            self.layers.append(hiddenLayer)
            self.layers.append(outputLayer)
        else:
            self.layers = layers

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        self.outputTask = outputTask  # Either classification or regression
        self.learningRate = learningRate
        self.epochs = epochs

    def getLayer(self, layerIndex):
        return self.layers[layerIndex]

    def getInputLayer(self):
        return self.getLayer(0)

    def getOutputLayer(self):
        return self.getLayer(-1)

    def feedForward(self, input):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        input : ndarray
            a numpy array (1,nIn + 1) containing the input of the layer

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the output layer
        """
        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        pass

    def computeError(self, target):
        """
        Compute the total error of the network

        Parameters
        ----------

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        pass

    def updateWeights(self):
        """
        Update the weights of the layers by propagating back the error
        """
        pass

    def train(self):
        # train procedures of the classifier

        # Initialize weights here for the input layer
        # If you use the default weight Initialization from Layer object,
        # you don't need to change this.

        pass

    def classify(self, testInstance):
        # classify an instance given the model of the classifier
        pass

    def evaluate(self, test):
        # evaluate a whole test set given the model of the classifier
        pass
