import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

def linear_discriminant_analysis(train_inputs, train_targets, validation_inputs, validation_targets, number_of_folds):

    accuracy = []
    for i in range(number_of_folds):
                lda = LinearDiscriminantAnalysis(solver = 'eigen')
                lda.fit(train_inputs[i], train_targets[i])
                predict_targets = lda.predict(validation_inputs[i])
                accuracy_i = accuracy_score(validation_targets[i], predict_targets)
                accuracy.append(accuracy_i)

    return lda

import matplotlib.pyplot as plt

def plotting(hello):

    scatter = plt.plot(hello) 

    return scatter