import warnings
warnings.simplefilter(action='ignore', category = FutureWarning)

import pandas as pd
import numpy as np
import seaborn as sns # visualization
import matplotlib.pyplot as plt
from mlp import MLP
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from numpy.random import seed


def covertWineClass(classCode):
    if (classCode == 1):
        return [1,0,0]
    elif (classCode == 2):
        return [0,1,0]
    elif (classCode == 3):
        return [0,0,1]

def getDataFrame(filePath, skipNum = 18, sepChars = ',\s'):
    df1 = pd.read_table(filePath, header=None, skiprows=skipNum, sep=sepChars, engine='python')
    result = [covertWineClass(c) for c in df1[13]]
    df2 = pd.DataFrame(result)
    df3 = pd.concat([df1, df2], axis=1, ignore_index=True)
    df3 = df3.iloc[np.random.permutation(len(df3))]
    return df3

dfs = []
for i in range(1, 11):
    df_tra = getDataFrame('wine/10-fold/wine-10-%dtra.dat' % i)
    df_tst = getDataFrame('wine/10-fold/wine-10-%dtst.dat' % i)

    target = df_tra.iloc[:,[14, 15, 16]]
    df_tra_, df_val_ , df_tra__, df_val__ = train_test_split(df_tra, target, train_size=142, test_size=18, random_state=0, stratify=target)

    df_tra = pd.concat([df_tra_, df_tra__], axis=1, ignore_index=True)
    df_val = pd.concat([df_val_, df_val__], axis=1, ignore_index=True)

    dfs.append([df_tra, df_tst, df_val])


seed(1)

eta = 0.1
precision = 10**-6
mom = 0
maxEpochs = 600

# topology
inputSize = 13
outputSize = 3

def run(topologiesVec):

    for layers in topologiesVec:

        topologyDesc = '-'.join([str(i) for i in (inputSize, *layers, outputSize)])

        results = []

        initialWeightsArr = [ [0]*3 for i in range(10)]

        for i in range(10):

            # Define the scaler for normalization
            scaler = StandardScaler().fit(dfs[i][0].iloc[:, 0:13])

            X = scaler.transform(dfs[i][0].iloc[:, 0:13])
            # X = dfs[i][0].iloc[:, 0:13].values
            y = dfs[i][0].iloc[:, [14, 15, 16]].values

            X_t = scaler.transform(dfs[i][2].iloc[:, 0:13])
            # X_t = dfs[i][2].iloc[:, 0:13].values
            y_t = dfs[i][2].iloc[:, [14, 15, 16]].values

            for j in range(0, 3):

                clf, w_, w, b, errors, epochs, elapsedTime = MLP(inputSize, outputSize, layers, maxEpochs, eta, mom, precision).fit(X, y)
                predictions = clf.predict(X_t, y_t, w, b)
                accuracy = metrics.accuracy_score(predictions, y_t)

                print('\n The prediction\'s accuracy is about: %0.2f' % accuracy)

                results.append([i+1, j+1, errors[-1], epochs[-1], elapsedTime, accuracy])

                initialWeightsArr[i][j] = w_

        df = pd.DataFrame(results, columns=['Fold', 'Execution', 'MSE', 'Epochs', 'Time', 'Accuracy'])
        df.to_csv('wine-executions/executions_%s.csv' % topologyDesc, sep = ';')
        df.describe().to_csv('wine-executions/executions_described_%s.csv' % topologyDesc, sep = ';')

        np.save('wine-weights/ini_weights_%s.npy' % topologyDesc, initialWeightsArr)

t = [(7,), (4,), (4,4), (12,)]

# Validates the chosen topologies
run(t)