from .learner import *
from .dataset import *
from .measurements import *

import argparse
import time
import random

class MLSystemManager(object):

    def get_learner(self, model):
        modelmap = {
            "baseline": BaselineLearner,
        }
        if model in modelmap:
            return modelmap[model]()
        else:
            raise Exception("Unrecognized model: {}".format(model))

    def load_dataset(self, file):
        return Dataset(file)

    def separate_targets(self, datasets):
        features = [dataset.inputs for dataset in datasets]
        targets = [dataset.targets for dataset in datasets]
        return features, targets



def get_args():
    parser = argparse.ArgumentParser(
        description="CS 478 Toolkit Command Line Module")

    parser.add_argument('learner',
                        metavar=['--learner', '-l', '-L'],
                        required=True,
                        choices=['baseline', 'perceptron', 'neuralnet',
                                 'decisiontree', 'knn'],
                        help='Choice of learning algorithm')

    parser.add_argument('file',
                        metavar=['--arff', '-a', '-A'],
                        required=True,
                        help='ARFF formatted dataset file.')

    parser.add_argument('eval_method',
                        metavar=['-e', '-E'],
                        required=True,
                        nargs='+',
                        help='Evaluation method.')

    # split: 0 or more args
    # default: just use the whole training set.
    # 1 arg: num folds (cross validation)
    # 2 args: %training %validation
    # 3 args: %training %validation %test
    parser.add_argument('--split', '-s',
                        type=float,
                        nargs='+')

    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help="Print metrics on individual class values.")

    parser.add_argument('--normalize', '-n',
                        action='store_true',
                        help='')

    parser.add_argument('--seed', '-r',
                        type=str,
                        help='specify a particular random seed')

    return parser.parse_args()


if __name__ == "main":
    args = get_args()

    manager = MLSystemManager()

    model = manager.get_learner(args.learner)

    dataset = manager.load_dataset(args.file)
    if args.normalize:
        dataset.normalize()

    print(f'Dataset Name: {dataset.name}')
    print(f'Dataset Size: {dataset.size}')
    print(f'Number of Attributes: {dataset.num_attributes}')
    print(f'Learning Algorithm: {args.learner}')
    print(f'Evaluation Method(s):' + ", ".join(args.eval_method))

    if not args.split:
        # No dataset split
        pass

    elif len(args.split) == 1:
        # Do crossfold validation
        splits = [1/args.split[0] for _ in range(args.split[0])]
        datasets = dataset.split(splits)
        feature_folds, target_folds = manager.separate_targets(datasets)

    elif len(args.split) == 2:
        # training/testing
        datasets = dataset.split(args.split)
        features, targets = manager.separate_targets(datasets)
        train_features, test_features = features
        train_targets, test_targets = targets

    elif len(args.split) == 3:
        # training/validation/testing
        datasets = dataset.split(args.split)
        features, targets = manager.separate_targets(datasets)
        train_features, val_features, test_features = features
        train_targets, val_targets, test_targets = targets


    switch = {
        'training-sse':,
        'training-mse':,
        'training-rmse':,
        'training-confusion':,
        'training-categorical':,
        'sse':,
        'mse':,
        'rmse':,
        'confusion':,
        'categorical':,
    }
