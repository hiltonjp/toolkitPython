from toolkit.supervised_learner import SupervisedLearner
import numpy as np

class Perceptron(SupervisedLearner):
    def __init__(self):
        self.weights = np.zeros(3)
        self.epochs = 20
        self.lr = 0.1

    def train(self, features, labels):
        features = np.array(features.data)
        bias = np.ones((features.shape[0], 1))
        print(features.shape)
        print(bias.shape)

        features = np.concatenate((features, bias), axis=1)

        labels = np.array(labels.data)

        print("Training")
        for e in range(self.epochs):
            print(f"Epoch {e+1}:")

            for i, (feat, targ) in enumerate(zip(features, labels)):
                net = np.dot(self.weights, feat)
                out = 1 if net > 0 else 0
                dW = self.lr*(targ-out)*feat
                self.weights += dW

                print(f'inputs-{i}: {feat}')
                print(f'net-{i}: {net}')
                print(f'out-{i}: {out}')
                print(f'targ-{i}: {targ}')
                print(f'dW-{i}: {dW}')
                print(f'new w: {self.weights}')
                print()

            print('\n\n')

    def predict(self, features, labels):
        features = np.array(features)

        bias = np.ones(1)
        features = np.concatenate((features, bias), axis=0)

        labels.clear()

        result = np.dot(self.weights, features)
        result = np.asscalar(result)
        result = result > 0

        labels.append(result)

    def _convert_features(self, features):
        pass

