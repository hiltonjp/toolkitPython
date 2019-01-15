import math
import numpy as np


def sse(model, features, labels):
    """Measure the sum squared error of a trained model on the given data."""

    if len(features) != len(labels):
        raise Exception \
            ("Expected the features and labels to have the same number of rows")
    if labels.num_attributes != 1:
        raise Exception \
           ("Sorry, this method only supports scalar labels")
    if features.size == 0:
        raise Exception("Expected at least one row")

    sse = 0
    for feat, target in zip(features, labels):
        pred = model.predict(feat)
        sse += (target-pred)**2

    return sse


def mse(model, features, labels):
    """Calculates the mean squared error."""
    return sse(model, features, labels)/features.size


def rmse(model, features, labels):
    """Calculates the root mean squared error."""
    return math.sqrt(mse(model, features, labels))


def confusion_matrix(model, features, labels):
    mat = np.zeros((labels.size, labels.size))

    for feat, target in zip(features, labels):
        pred = model.predict(feat)
        mat[int(target), int(pred)] += 1

    # normalization
    exp = np.exp(mat-np.max(mat))
    mat = exp/exp.sum(axis=0)
    return mat


def categorical_accuracy(model, features, labels):
    """Calculate a raw accuracy: num correct/total"""
    correct = 0

    for feat, target in zip(features, labels):
        pred = model.predict(feat)
        correct += 1 if int(round(pred)) == int(target) else 0

    return correct/len(labels)
