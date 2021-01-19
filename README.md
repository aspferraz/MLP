# MLP

Implementation of Multilayer Perceptron Networks (MLP) in Python with examples of problem solving
three classification problems: Iris Plants, Glass identification and White Wine Quality. These
data sets can be downloaded from the Keel Dataset repository (https://sci2s.ugr.es/keel/datasets.php).

The best topology is chosen for each problem using 10-folds cross-validation. Each fold is performed three times using the standard learning algorithm
backpropagation, initializing the weight matrices with random values between 0 and 1. The logistic activation function (sigmoid) is used for all neurons,
learning = 0.1 and precision = 10 ^ -6. For each topology tested, a table containing the mean and standard deviation of the following measures: NDE, Number of epochs, Time (seconds) and Accuracy (percentage of correctness of the validation set) is created.

After finding the best topology for each problem, the algorithm is performed three times each fold through
of the backpropagation learning algorithm with momentum.

Finally, the networks are tested using the test sets for each problem. A table is created with the
average and standard deviation of the hit rates (%) in each problem.
