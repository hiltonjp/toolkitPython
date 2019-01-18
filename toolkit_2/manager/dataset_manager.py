from .dataset import Dataset


class DatasetManager(object):
    def __init__(self):
        super().__init__()

    def load_dataset(self, file):
        return Dataset(file)

    def save_dataset(self, dataset, file):
        with open(file) as f:
            f.write(str(dataset))
            f.close()

    def split(self, dataset, split=0.5):
        return dataset.split(split)

    def merge(self, datasets):
        return sum(datasets)

    def shuffle(self, dataset):
        dataset.shuffle()
        return dataset

    def normalize(self, dataset):
        dataset.normalize()
        return dataset

