from .dataset import Dataset


class DatasetManager(object):

    def __init__(self):
        super().__init__()

    @classmethod
    def load_dataset(cls, path):
        """Load an ARFF dataset."""
        return Dataset(path)

    @classmethod
    def save_dataset(cls, dataset, path):
        """Save a Dataset out to an ARFF file.

        TODO: test. You should be able to load the same dataset back into memory
        """
        with open(path) as f:
            f.write(str(dataset))
            f.close()

    @classmethod
    def split(cls, dataset, split=0.5):
        """Split the data into separate folds.

        Args:
        split (float | list[float...]):
            - if the argument is a float, will split the dataset in half with
              the first half getting the percentage indicated.
              Expected values: [0, 1]

            - if the argument is a list, it will split the dataset into
              len(split) separate Datasets according to the percentages listed.
              Expected values in list: [0, 1]

        return (list[Dataset...]): a list of Datasets, in the order given by
            split argument.


        TODO: test
        """
        return dataset.split(split)

    @classmethod
    def merge(cls, datasets):
        """Merges a list of Datasets into a single Dataset."""
        return sum(datasets)

    @classmethod
    def shuffle(cls, dataset):
        """Randomly shuffles a dataset's instances."""
        dataset.shuffle()
        return dataset

    @classmethod
    def normalize(cls, dataset):
        """Normalizes a dataset's instance values to [0, 1]."""
        dataset.normalize()
        return dataset

