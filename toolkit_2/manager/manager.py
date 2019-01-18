from .dataset import *
from .measurements import *
from .states import *

from ..learners import *

import matplotlib.pyplot as plt

class MLSystemManager(object):
    def __init__(self):
        self.state = NoValidation()

        # Options for evaluating a model
        self.metrics = {
            'sse':(sse,"Sum Squared Error"),
            'mse':(mse, "Mean Squared Error"),
            'rmse':(rmse, "Root Mean Squared Error"),
            'confusion':(confusion_matrix, "Confusion Matrix"),
            'categorical':(categorical_accuracy, "Raw Accuracy")
        }

        # Options of model
        self.models = {
            'baseline':BaselineLearner,
            'perceptron':Perceptron,
            'neuralnet':NeuralNet,
            'knn':KNN,
            'decisiontree':DecisionTree
        }

    def get_learner(self,model):
        if model in self.models:
            return self.models[model]()
        else:
            raise Exception("Unrecognized model: {}".format(model))

    def gather_metrics(self, methods):
        """Gather up the types of metrics the user wants for their model."""
        return [self.metrics[method] for method in methods]


    ############################################################################
    # DATASET MANAGEMENT INTERFACE                                             #
    ############################################################################

    def load_dataset(self, file):
        return Dataset(file)

    def split(self, dataset, split):
        return self.state.split(dataset, split)

    ############################################################################
    # STATE INTERFACE                                                          #
    ############################################################################

    def do_validation(self):
        self.state = WithValidation()

    def cross_validation(self):
        self.state = CrossValidation()

    def no_validation(self):
        self.state = NoValidation()

    def no_split(self):
        self.state = NoSplit()

    ############################################################################
    # TRAINING/TESTING                                                         #
    ############################################################################

    def describe(self, dataset, learner_name, eval_methods):
        print(f'Dataset Name: {dataset.name}')
        print(f'Dataset Size: {dataset.size}')
        print(f'Number of Attributes: {dataset.num_attributes}')
        print(f'Learning Algorithm: {learner_name}')
        print(f'Evaluation Method(s):' + ", ".join(eval_methods))
        print()

    def train(self, model, features, targets):
        self.state.train(model, features, targets)

    def test(self, model, features, targets, methods):
        self.state.test(model, features, targets, methods)

    def display(self, metrics):
        for name in metrics:
            if name is "Confusion Matrix":
                mat = metrics[name]
                plt.title(name)
                plt.xticks(range(mat.shape[0]), )
                plt.imshow(mat)
                plt.show()
