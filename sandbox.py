import numpy as np
import pandas as pd

# activation func
def sigmoid(u, b=0.5):
    return 1. / (1 + np.exp(-b * u))

def sigmoidDerivative(u, b=0.5):
    return b * sigmoid(u) * (1 - sigmoid(u))

def getNormalizedOutput(y):
    if y >= 0.5:
        return 1
    else:
        return 0

def getFinalError(d, y):
    sum = 0
    for j in range(len(y)):
        sum = sum + (d[j] - getNormalizedOutput(y[j][1]))**2

    return 1./2 * sum
eta = 0.1
inputs = 4
outputs = 3
layers = (6, 4)
weights = []
for l in range(len(layers)):
    if (l == 0):
        weights.append(np.transpose( np.random.uniform(0, 1, ( inputs, layers[l]) ) ) )
    if (l < (len(layers) - 1)):
        weights.append(np.transpose( np.random.uniform(0, 1, (layers[l], layers[l+1]) ) ) )
    else:
        weights.append(np.transpose( np.random.uniform(0, 1, (layers[l], outputs) ) ) )

activations = []
for l in range(len(layers)):
    # activations.append([np.zeros(layers[l]), np.zeros(layers[l])])
    activations.append([[np.random.uniform(0,0.9) for _ in range(layers[l])], [np.random.uniform(0,0.9) for _ in range(layers[l])]])


outLayerActivations = [0]*outputs
for ol in range(outputs):
    # print(activations[-1][1])
    # print(iris-weights[-1][ol])
    pre = np.dot(activations[-1][1], weights[-1][ol])
    # print(pre)
    post = sigmoid(pre)
    # print(post)
    outLayerActivations[ol] = [pre, post]

# print(outLayerActivations)

 #2.2: Compute the output layer's error
Y = [0, 1, 0]
errors = []
for n in range(outputs):
    y = outLayerActivations[n][1]
    # deltaOut += ( (y - Y[n]) * sigmoidDerivative(outLayerActivations[n][1]) ) ** 2
    errors.append( ( (y - Y[n])**2 ) * sigmoidDerivative(outLayerActivations[n][1]))
deltaOut = (1/2)*sum(errors)
print (deltaOut)

a = [[0.01895276, 0.03015534, 0.0613778,  0.0260315,  0.01000285, 0.04092527,
  0.0003963,  0.02707075],
 [0.01812982, 0.03549912, 0.00670976, 0.02066392, 0.05383293, 0.03177738,
  0.00398775, 0.02953082],
 [0.02222084, 0.01786938, 0.01738915, 0.0578475,  0.00918505, 0.039417,
  0.0506866, 0.006425  ],
 [0.05809377, 0.03476727, 0.0067234,  0.00445595, 0.05332672, 0.04056107,
  0.06173761, 0.03117363]]
b = [0.1199423,  0.12009572, 0.11982998, 0.11902922]

# print (np.dot(np.transpose(a),b))
# for i in reversed(range(len(layers))):
#     print(i)


# import collections
# x = collections.deque(2*[None], 2)
# print(x)
# x.appendleft(1)
# print(x)
# x.appendleft(2)
# print(x)
# x.appendleft(3)
# print (x)
#
# from numpy.random import seed
# from numpy.random import rand
# # seed random number generator
# seed(1)
# # generate some random numbers
# print(rand())
# # reset the seed
#
# # generate some random numbers
# print(rand())
#
# # seed(1)
# print(rand())

# weightsArr = [ [0]*3 for i in range(10)]
# print(weightsArr)


# w = np.asarray([[1,2],[1,2,3]])

t = [(3,3), (4,), (2,)]
for layers in t:
    print(layers)