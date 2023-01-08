import sklearn
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from fomlads.evaluate.eval_regression import train_and_test_partition
from fomlads.evaluate.eval_regression import train_and_test_filter
from fomlads.evaluate.eval_regression import create_cv_folds


# LOGISTIC REGRESSION

# Perform cross-validation for each hyperparameter value on training + validation data


def logistic_regression_best_lamda(train_inputs, train_targets, validation_inputs, validation_targets, number_of_folds):

    powers_of_ten = np.linspace(-4, 5, 10)
    ten = np.full(10, 10)
    list_of_parameters = list(np.power(ten, powers_of_ten, dtype = float))

    mean_accuracies = []

    for parameter in list_of_parameters:
        accuracy_j = []
        for i in range(number_of_folds):
            LogReg = LogisticRegression(C = parameter, max_iter = 10000)
            LogReg.fit(train_inputs[i], train_targets[i])
            predict_targets = LogReg.predict(validation_inputs[i])
            accuracy_for_fold_i = accuracy_score(validation_targets[i], predict_targets)
            accuracy_j.append(accuracy_for_fold_i)
        mean_accuracy_j = np.mean(accuracy_j)
        mean_accuracies.append(mean_accuracy_j)


    # Choose the best hyperparameter value averaged over folds by finding the highest 
    # mean accuracy

    mean_accuracies_list = np.array(mean_accuracies)
    index_of_max_mean = mean_accuracies_list.argmax()
    best_lambda = list_of_parameters[index_of_max_mean]

    return best_lambda


def test_log_reg(train_validation_inputs,train_validation_targets, best_lambda):
    LogReg2 = LogisticRegression(C = best_lambda, max_iter = 10000)
    LogReg2.fit(train_validation_inputs, train_validation_targets)

    return LogReg2