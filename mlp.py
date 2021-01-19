'''
    File name: mlp.py
    Author: Antonio Ferraz
    Date created: 09/26/2020
    Date last modified: 09/30/2020
    Python Version: 3.6
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy.random import rand
import collections
import copy
import time

class MLP(object):

    def __init__(self, inputs, outputs, layers, epochs = 600, eta = 0.1, mom = 0, precision = 10**-6):
        self.eta = eta # learning rate
        self.mom = mom # momentum factor
        self.precision = precision # exit condition
        self.epochs = epochs # max iters

        # topology
        self.inputs = inputs
        self.layers = layers
        self.outputs = outputs

    def backPropagation(self, x, z, y, w, b, phi, mom = 0, pWeights = None, pBiases = None):
        errorOut = y - z[-1]
        deltaOut = -1 * errorOut * phi(z[-1])
        deltasHdd = [0] * (len(self.layers) + 1)

        pWeights_ = []
        for i in range(len(w)):
            pWeights_.append(np.zeros_like(w[i]))

        previousWeights = pWeights_ if not pWeights else pWeights

        pBiases_ = []
        for i in range(len(b)):
            pBiases_.append(np.zeros_like(b[i]))

        previousBiases = pBiases_ if not pBiases else pBiases


        for l in reversed(range(len(self.layers) + 1)):

            if (len(self.layers) == l):
                for i in range(self.layers[-1]):
                    for j in range(self.outputs):
                        w[-1][i][j] -= self.eta * (deltaOut[j] * z[-2][i]) + ( mom * (w[-1][i][j] - previousWeights[-1][i][j]) )
                        b[-1][j] -= self.eta * deltaOut[j] + ( mom * (b[-1][j] - previousBiases[-1][j]) )

                deltasHdd[l] =  np.matmul(w[-1], deltaOut) \
                           * phi(z[-2])

            if (l == 0):
                inputs = len(x)
                delta = deltasHdd[l + 1]

                for i in range(inputs):
                    for j in range(self.layers[l]):
                        w[l][i][j] -= self.eta * (delta[j] * x[i]) + ( mom * (w[l][i][j] - previousWeights[l][i][j]) )
                        b[l][j] -= self.eta * delta[j] + ( mom * (b[l][j] - previousBiases[l][j]) )

            else:
                if (l < len(self.layers)):
                    inputs = self.layers[l - 1]
                    delta = deltasHdd[l + 1]

                    for i in range(inputs):
                        for j in range(self.layers[l]):
                            w[l][i][j] -= self.eta * (delta[j] * z[l-1][i]) + ( mom * (w[l][i][j] - previousWeights[l][i][j]) )
                            b[l][j] -= self.eta * delta[j] + ( mom * (b[l][j] - previousBiases[l][j]) )

                    deltasHdd[l] = np.matmul(w[l], delta) * phi(z[l-1])

        return w, b

    def fit(self, X, d, w = None):

        if w is None:
            w = []

        start_time = time.time()

        # seed(1)

        weights = []
        biases = []

        for l in range(len(self.layers)):
            if (l == 0):
                previousLayer = X.shape[1]
            else:
                previousLayer = self.layers[l - 1]


            weights.append([[rand() for i in
                             range(self.layers[l])] for j in
                             range(previousLayer)]) if len(w) == 0 else weights.append(w[l])

            biases.append(np.array([-1 for i in
                                    range(self.layers[l])]))


        weights.append([[rand() for i in
                         range(self.outputs)] for j in
                         range(self.layers[-1])]) if len(w) == 0 else weights.append(w[-1])

        biases.append(np.array([-1 for i in
                                    range(self.outputs)]))

        initialWeights = copy.deepcopy(weights) # makes an copy of the original weights

        totalErrors = 0
        errorsVec = [] # errors vector by epoch
        epochsVec = []

        twoLastWeights = collections.deque(2 * [None], 2)
        twoLastBiases = collections.deque(2 * [None], 2)

        for epoch in range(1, self.epochs):

            for sample in range(0, X.shape[0]):

                x = X[sample]

                # Feed forward
                z = []
                for l in range(len(self.layers)):
                    if (l == 0):
                        i = x
                    else:
                        i = z[l - 1]

                    z.append(self.sigmoid(np.dot(i, weights[l]) + biases[l].T))  # output layer l

                y = self.sigmoid(np.dot(z[-1], weights[-1]) + biases[-1].T)  # output last layer
                z.append(y)

                output = d[sample]

                # Backward
                twoLastWeights.appendleft(copy.deepcopy(weights))
                twoLastBiases.appendleft(copy.deepcopy(biases))

                weights, biases = self.backPropagation(x, z, output, weights, biases, self.sigmoidDeriv, self.mom, twoLastWeights[1], twoLastBiases[1])

                squareError = 0
                for i in range(len(output)):
                    error = (output[i] - y[i]) ** 2
                    squareError = squareError + 0.5 * error
                    totalErrors = totalErrors + squareError

            # MSE
            totalErrors = totalErrors / X.shape[0]
            mse = totalErrors

            if epoch % 25 == 0 or epoch == 1:
                print('Epoch ', epoch, '- Error: ',
                      mse)

            if (len(errorsVec) > 0):
                if (abs(mse - errorsVec[-1]) <= self.precision):
                    print('Pattern learned in ' + str(epoch) + ' epochs')
                    print('Mean Squared Error: ', str(mse))
                    break

            errorsVec.append(mse)
            epochsVec.append(epoch)

        print('Initial Weights: ', initialWeights)
        print('Final Weights: ', weights)

        # self.showErrorGraphic(errorsVec, epochsVec)

        return self, initialWeights, weights, biases, errorsVec, epochsVec, round(time.time() - start_time, 2)

    def showErrorGraphic(self, errors, epochs):
        plt.figure(figsize=(9, 4))
        plt.plot(errors, epochs, 'm-', color='b')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Mean Squared Error (MSE) ')
        plt.title('Error Minimization')
        plt.show()

    def predict(self, X, y, weights, biases):

        predictions = []

        for sample in range(0, X.shape[0]):  # loop in all the passengers
            x = X[sample]

            # Feed forward
            z = []
            for l in range(len(self.layers)):
                if (l == 0):
                    i = x
                else:
                    i = z[l - 1]

                z.append(self.sigmoid(np.dot(i, weights[l]) + biases[l].T))  # output layer l

            y_ = self.sigmoid(np.dot(z[-1], weights[-1]) + biases[-1].T)  # last layer

            # 0 if y < 0.5, else 1
            predictions.append(np.heaviside(y_ - 0.5, 1))


        # print('\n Number of Sample  | Class              | Expected Output   | Prediction   ')
        # for i in range(y.shape[0]):
        #
        #     if (y[i] == np.array([1, 0, 0])).all() :
        #         print(
        #             ' id :',
        #             '{:03.0f}'.format(i),
        #             '         | 1                  |',
        #             [1, 0, 0],
        #             '        |',
        #             predictions[i],
        #         )
        #     elif (y[i] == np.array([0, 1, 0])).all() :
        #         print(
        #             ' id :',
        #             '{:03.0f}'.format(i),
        #             '         | 2                  |',
        #             [0, 1, 0],
        #             '        |',
        #             predictions[i],
        #         )
        #     elif (y[i] == np.array([0, 0, 1])).all() :
        #         print(
        #             ' id :',
        #             '{:03.0f}'.format(i),
        #             '         | 3                  |',
        #             [0, 0, 1],
        #             '        |',
        #             predictions[i],
        #         )

        return predictions



    # Logistic Sigmoid
    def sigmoid(self, u, b = 1):
        from scipy.special import expit
        return expit(b * u) # More robust
        # return 1 / (1 + np.exp(b * - u))

    def sigmoidDeriv(self, u, b = 1):
        # g = self.sigmoid(u)

        # Assumes that u is already calculated using the sigmoid function,
        # and so it is not to be re-computed the second time.
        g = u
        return b * g * (1 - g)

    # Rectifier Linear Unit (ReLU)
    def relu(self, u, b = 1):
        return np.maximum(b * u, 0)

    def reluDeriv(self, u, b = 1):
        return np.heaviside(b * u, 1)

    # Hyperbolic Tangent
    def tanh(self, u, b = 1):
        return np.tanh(b * u)

    def tanhDeriv(self, u, b = 1):
        return 1 - (b * u) ** 2