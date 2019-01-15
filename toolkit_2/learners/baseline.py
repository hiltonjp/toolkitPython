from .learner import *

import numpy as np
import scipy.stats as stats

class BaselineLearner(SupervisedLearner):

    def __init__(self):
        super().__init__()
        self.labels = []

    def train(self, features, labels):
        self.labels.clear()
        for i in range(labels.num_attributes):
            col = labels.get_attribute_column(i)
            if labels.is_continuous(i):
                self.labels.append(np.mean(col))
            else:
                self.labels.append(stats.mode(col, axis=None))

    def predict(self, features):
        pred = self.labels
        return pred