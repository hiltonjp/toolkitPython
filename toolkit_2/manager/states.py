import time


class ManagerState(object):
    """ManagerState Abstract Class

    """

    def __init__(self):
        self.start = 0
        self.end = 0

    def split(self, dataset, splits):
        raise NotImplementedError

    def train(self, model, features, targets):
        raise NotImplementedError

    def test(self, model, features, targets, methods):
        raise NotImplementedError


    def start_timer(self):
        self.start = time.time()

    def end_timer(self):
        self.end = time.time()
        print(f"Time to train: {self.end - self.start}s")

    @staticmethod
    def separate_targets(datasets):
        features = [dataset.inputs for dataset in datasets]
        targets = [dataset.targets for dataset in datasets]
        return features, targets


class NoSplit(ManagerState):
    """NoSplit()
    Make no split of the dataset into Training/Test.
    """

    def split(self, dataset, splits):
        features, targets = self.separate_targets([dataset])
        return features, targets

    def train(self, model, features, targets):
        self.start_timer()
        model.train(features, targets)
        self.end_timer()

    def test(self, model, features, targets, methods):
        metrics = {}
        for method in methods:
            method, name = method
            metrics[name] = method(model, features, targets)
        return metrics


class NoValidation(ManagerState):
    """NoValidation()
    Separate dataset into Training/Testing sets.
    """

    def split(self, dataset, splits):
        datasets = dataset.split(splits)
        features, targets = self.separate_targets(datasets)
        return features, targets

    def train(self, model, features, targets):
        training_feats = features[0]
        training_targs = targets[0]

        self.start_timer()
        model.train(training_feats, training_targs)
        self.end_timer()

    def test(self, model, features, targets, methods):
        features = features[1]
        targets = targets[1]
        metrics = {}
        for method in methods:
            method, name = method
            metrics[name] = method(model, features, targets)
        return metrics

class WithValidation(ManagerState):
    """WithValidation()

    """
    #TODO implement

    def split(self, dataset, splits):
        datasets = dataset.split(splits)
        features, targets = self.separate_targets(datasets)
        return features, targets

    def train(self, model, features, targets):
        pass

    def test(self, model, features, targets, methods):
        pass


class CrossValidation(ManagerState):
    """CrossValidation()

    """
    #TODO implement

    def split(self, dataset, splits):
        split = splits[0]
        splits = [1/split for _ in range(split)]

        datasets = dataset.split(splits)
        features, targets = self.separate_targets(datasets)
        return features, targets

    def train(self, model, features, targets):
        training_losses = []
        validation_losses = []

    def test(self, model, features, targets, methods):
        pass
