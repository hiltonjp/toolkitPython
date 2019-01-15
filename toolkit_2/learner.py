class SupervisedLearner(object):

    def train(self, features, labels):
        """
        Before you call this method, you need to divide your data
        into a feature matrix and a label matrix.
        :type features: Matrix
        :type labels: Matrix
        """
        raise NotImplementedError()

    def predict(self, features):
        """
        A feature vector goes in. A label vector comes out. (Some supervised
        learning algorithms only support one-dimensional label vectors. Some
        support multi-dimensional label vectors.)
        :type features: [float]
        :type labels: [float]
        """
        raise NotImplementedError()


