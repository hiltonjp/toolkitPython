from .manager import MLSystemManager

import argparse


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

    parser.add_argument('eval_methods',
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

    # Configure training/validation/testing split.
    if not args.split:
        manager.no_split()
    elif len(args.split) == 1:
        manager.cross_validation()
    elif len(args.split) == 2:
        manager.no_validation()
    elif len(args.split) == 3:
        manager.do_validation()

    # Print description of your basic configuration.
    manager.describe(dataset, args.learner, args.eval_method)

    # Do training.
    features, targets = manager.split(dataset, args.split)
    manager.train(model, features, targets)

    # Do model evaluation
    metrics = manager.gather_metrics(args.eval_methods)
    manager.test(model, features, targets, metrics)

