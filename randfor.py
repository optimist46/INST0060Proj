import sklearn
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from fomlads.evaluate.eval_regression import train_and_test_partition
from fomlads.evaluate.eval_regression import train_and_test_filter
from fomlads.evaluate.eval_regression import create_cv_folds




def random_forest_n_estimator(train_inputs,train_targets, validation_inputs, validation_targets, number_of_folds):
    # Here we only discuss the parameter: n_estimator
    # n_estimator: number of the tree (from 1 to 50)
    num_tree = 50
    list_n_estimators = [n+1 for n in range(num_tree)]
    mean_accuracies_rf = []

    for parameter in list_n_estimators:
        accuracy_fold = []
        for i in range(number_of_folds):
            Mod_rf = RandomForestClassifier(n_estimators=parameter)
            Mod_rf.fit(train_inputs[i], train_targets[i])
            predict_targets = Mod_rf.predict(validation_inputs[i])
            accuracy_for_fold_i = accuracy_score(validation_targets[i], predict_targets)
            accuracy_fold.append(accuracy_for_fold_i)
        mean_accuracy_fold = np.mean(accuracy_fold)
        mean_accuracies_rf.append(mean_accuracy_fold)

    best_index = np.argmax(np.array(mean_accuracies_rf))
    best_n_estimator = best_index + 1

    return best_n_estimator


def test_rand_for(train_validation_inputs, train_validation_targets, best_n_estimator):
    Mod_rf2 = RandomForestClassifier(n_estimators=best_n_estimator)
    Mod_rf2.fit(train_validation_inputs, train_validation_targets) 

    return Mod_rf2