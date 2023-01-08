from fomlads.evaluate.eval_regression import train_and_test_partition
from fomlads.evaluate.eval_regression import train_and_test_filter
from fomlads.evaluate.eval_regression import create_cv_folds

import numpy as np
import pandas as pd

# import_data function imports a given file into a pandas dataset.
    # it takes a variable containing the filepath

def import_data(filepath = 'Data/New1_noQuestionMark.csv'):
    dataset = pd.read_csv(filepath)
    return dataset


# inputs_and_targets is a function separates inputs and targets in the dataset.
    # it takes a pandas dataset as input

def inputs_and_targets(pdDataset):
    number_of_columns = len(pdDataset.axes[1])
    npDataset = pdDataset.to_numpy() # Convert dataset to a numpy array
    inputs = npDataset[: , :(number_of_columns - 1)] # Split numpy array since genres/targets are only in the last column
    targets = npDataset[: , (number_of_columns - 1)].astype('object')
    return inputs, targets


# train_and_validation_inputs_targets function 
# takes dataset for the training/validation data, then creates n (= cv_folds) cross validation folds to use in different models.
    

def train_and_validation_inputs_targets(inputs,targets, cv_folds = 5, randomseed = 1):
    np.random.seed(randomseed) # Use a random seed = 0 to ensure reproducability when creating cross validation folds

    folds = create_cv_folds(len(inputs), cv_folds) # Get n (= cv_folds) train and test filters from create_cv_folds to use cross validation
    
    train_inputs = []
    train_targets = []
    validation_inputs = []
    validation_targets = []
    
    for i in range(cv_folds): # Store train and validation inputs/targets as lists
        train_inputs1, train_targets1, validation_inputs1, validation_targets1 = train_and_test_partition(inputs, targets, folds[i][0], folds[i][1])
        train_inputs.append(train_inputs1)
        train_targets.append(train_targets1)
        validation_inputs.append(validation_inputs1)
        validation_targets.append(validation_targets1)
        
    return train_inputs, train_targets, validation_inputs, validation_targets