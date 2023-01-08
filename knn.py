import sklearn
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from fomlads.evaluate.eval_regression import train_and_test_partition
from fomlads.evaluate.eval_regression import train_and_test_filter
from fomlads.evaluate.eval_regression import create_cv_folds

from sklearn.neighbors import KNeighborsClassifier



def knn_k_value(train_inputs, train_targets, validation_inputs, validation_targets, number_of_folds):
    mean_accuracies_knn = []
    k_value = [n for n in range(1,101)]

    for k in k_value:
        accuracy = []
        for i in range(number_of_folds):
            Knn = KNeighborsClassifier(n_neighbors = k)
            Knn.fit(train_inputs[i], train_targets[i])
            predict_targets = Knn.predict(validation_inputs[i])
            accuracy_i = accuracy_score(validation_targets[i], predict_targets)
            accuracy.append(accuracy_i)
        mean_accuracy_fold = np.mean(accuracy)
        mean_accuracies_knn.append(mean_accuracy_fold)

    best_knn_index = np.argmax(np.array(mean_accuracies_knn))
    best_k_value = best_knn_index+1

    return best_k_value

def test_knn(train_validation_inputs, train_validation_targets, best_k_value):
    Knn2 = KNeighborsClassifier(n_neighbors = best_k_value)
    Knn2.fit(train_validation_inputs, train_validation_targets)
    
    return Knn2