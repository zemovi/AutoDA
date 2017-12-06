#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
import numpy as np


def compute_zero_mean_unit_variance(X, mean=None, std=None):

    """ Compute zero mean and unit variance of training set
        for normalizating augmented batch in training

    Parameters
    ----------
    X : TODO

    Returns
    -------
    mean
    std
    """

    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    return mean, std


def normalize(X, mean, std):
    """ Normalize data by its zero mean and standard variance

    Parameters
    ----------
    batch_x

    Returns
    -------
    x_train_normalized
    x_test_normalized

    """

    X_normalized = (X - mean) / (std +  1e-7)

    return X_normalized
