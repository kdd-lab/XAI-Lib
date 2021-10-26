#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:15:54 2019

@author: francesco
"""
import numpy as np
from sklearn.metrics import accuracy_score


class DecodePredictWrapper(object):
    def __init__(self, decoder, blackbox):
        self.decoder = decoder
        self.blackbox = blackbox

    def predict(self, X):
        return self.blackbox.predict(self.decoder.predict(X))


class BlackboxWrapper(object):
    """Blackbox wrapper
    In this package any blackbox must take a 3d input [batch, n_timesteps, 1] and return a 1d  output [batch,
    ]  containing the labels. This wrapper converts automatically the input and output to the required shape.

    Parameters
    ----------
    blackbox : trained model that takes a datasets as input and outputs labels
    input_dimensions: int
        number of dimensions of the input data of the blackbox (2 or 3) 2: [batch, n_timesteps], 3: [batch,
        n_timesteps, 1]
    output_dimensions: int
        number of dimensions of the output labels of the blackbox (1 or 2) 1: [batch,] or [batch, 1],
        2: [batch, n_classes]
    
    Attributes
    ----------
    
    """

    def __init__(self, blackbox, input_dimensions, output_dimensions):
        self.blackbox = blackbox
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions

    def predict(self, X):
        """Wrapper for the predict method.
        Parameters
        ----------
        X : {array-like}
            Test data. Shape [batch, timesteps, 1].
        Returns
        -------
        y : array of shape [batch]
        """
        if len(X.shape) < 2:
            raise ValueError("Input data should have at least 2 dimensions")
        if self.input_dimensions == 2:
            X = X[:, :, 0]  # 3d to 2d array [batch, timesteps]
        y = self.blackbox.predict(X)
        if len(y.shape) > 1 and (y.shape[1] != 1):
            y = np.argmax(y, axis=1)  # from probability to  predicted class
        y = y.ravel()
        return y

    def predict_proba(self, X):
        """Wrapper for the predict_proba method.
        Parameters
        ----------
        X : {array-like}
            Test data. Shape [batch, timesteps, 1].
        Returns
        -------
        y : array of shape [batch, n_classes]
        """
        if len(X.shape) < 2:
            raise ValueError("Input data should have at least 2 dimensions")
        if self.input_dimensions == 2:
            X = X[:, :, 0]  # 3d to 2d array [batch, timesteps]

        if self.output_dimensions == 1:
            y = self.blackbox.predict_proba(X)
        else:
            y = self.blackbox.predict(X)

        return y

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


if __name__ == "__main__":
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = [0, 0, 1, 1]

    print("\nSCIKIT")
    from sklearn.neighbors import KNeighborsClassifier

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)

    print("predict", neigh.predict([[1.1, 2]]))
    print("predict_proba", neigh.predict_proba([[0.9, 1]]))

    print("\nWRAPPER SCIKIT")
    blackbox = BlackboxWrapper(neigh, 2, 1)
    print("predict", blackbox.predict(np.array([[1.1, 2]]).reshape(1, -1, 1)))
    print("predict_proba", blackbox.predict_proba(np.array([[0.9, 1]]).reshape(1, -1, 1)))
