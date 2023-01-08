import numpy as np
import pandas as pd

def fisher_lda(X, y):
    # Calculate the mean vectors for each class
    mean_vectors = []
    n_classes = 5
    n_features = len(X.iloc[0])


    for cl in range(n_classes):
        mean_vectors.append(np.mean(X[y==cl], axis=0))

    # Calculate the within-class scatter matrix
    S_W = np.zeros((n_features, n_features))
    for cl, mv in zip(range(n_classes), mean_vectors):
        class_scatter = np.cov(X[y==cl].T)
        S_W += class_scatter

    # Calculate the between-class scatter matrix
    overall_mean = np.mean(X, axis=0)
    S_B = np.zeros((n_features, n_features))
    for i, mean_vec in enumerate(mean_vectors):
        n = X[y==i+1, :].shape[0]
        mean_vec = mean_vec.reshape(n_features, 1)
        overall_mean = overall_mean.reshape(n_features, 1)
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

    # Solve the eigenvalue problem
    eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    # Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

    # Select the k eigenvectors with the largest eigenvalues
    W = np.hstack([eigen_pairs[i][1]])


    return W


def fisher_lda_one_vs_rest(X, y):
    # Create a list to store the models
    models = []
    n_components = len(X.iloc[0])

    # Loop over the different classes
    for cl in np.unique(y):
        # Create a binary class label
        y_binary = np.where(y == cl, 1, 0)

        # Calculate the mean vector for the binary class
        mean_vector = np.mean(X[y_binary==1], axis=0)

        # Calculate the within-class scatter matrix
        S_W = np.cov(X[y_binary==1].T)

        # Calculate the between-class scatter matrix
        overall_mean = np.mean(X, axis=0)
        S_B = mean_vector.dot(mean_vector.T)

        # Solve the eigenvalue problem
        eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

        # Make a list of (eigenvalue, eigenvector) tuples
        eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

        # Select the k eigenvectors with the largest eigenvalues
        W = np.hstack([eigen_pairs[i][1] for i in range(n_components)])

        # Transform the data using the eigenvectors
        X_transformed = X.dot(W)

        # Store the model
        models.append((W, overall_mean))

    return models
