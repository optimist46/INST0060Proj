from fomlads.evaluate.eval_regression import train_and_test_partition
from fomlads.evaluate.eval_regression import train_and_test_filter
from fomlads.evaluate.eval_regression import create_cv_folds

import matplotlib.pyplot as plt
import matplotlib.colors

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


def plot_decision_boundary(dataset,train_validation_inputs,train_validation_targets,test_inputs, model, feature1, feature2):
   

    # Assume that you have X (features) and y (labels) for training data, and X_test (features) for testing data
    # Also assuming that X has k features

    X = train_validation_inputs
    X_test = test_inputs
    y = train_validation_targets
    y_numarated = np.unique(y,return_inverse= True)[1]



    # Selecting two features
    X_2d = X[:, [feature1, feature2]] # select first two features

    

    # Fit the model to the training data
    model.fit(X_2d, y)

    # use the model to predict the labels of the test data
    predictions = model.predict(X_test[:, [0, 1]])

    # Get the min and max values for the features
    x_min, x_max = X_2d[:, 0].min() - 0.2, X_2d[:, 0].max() + 0.2
    y_min, y_max = X_2d[:, 1].min() - 0.2, X_2d[:, 1].max() + 0.2

    # Create a mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.03),
                        np.arange(y_min, y_max, 0.03))

    # Flatten the mesh grid
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    # Make predictions using the model
    predictions = model.predict(X_grid)

    # Reshape the predictions to the original shape
    predictions = predictions.reshape(xx.shape)
    predictions_numerated = np.unique(predictions,return_inverse= True)[1].reshape(predictions.shape)

    #color palette
    colors = ['Red','Blue','Green','Yellow','Purple']

    # Draw the decision boundaries
    plt.contourf(xx, yy, predictions_numerated, cmap=matplotlib.colors.ListedColormap(colors), alpha=0.5)



    # Plot the training points
    for i in np.unique(y):
        plt.scatter(X_2d[y==i, 0], X_2d[y==i, 1], c=y_numarated[y==i], cmap=matplotlib.colors.ListedColormap(colors[int(np.where(np.unique(y)==i)[0])]), s = 30, alpha = 0.6, label=i)
        
    plt.rcParams['figure.figsize'] = [15, 15]

    # Display the plot
    plt.title('{} and {}'.format(dataset.columns[feature1],dataset.columns[feature2]))
    #plt.gca().legend()
    plt.xlabel(dataset.columns[feature1])
    plt.ylabel(dataset.columns[feature2])
    plt.show()

    return plt