from unittest import TestCase

from toolkit_2.manager.dataset import Dataset

class DatasetTest(TestCase):

    file = '../test_data/test.arff'

    def test_load(self):
        dataset = Dataset(self.file)

        self.assertTrue(dataset.name == "test")
        self.assertTrue(dataset.size == 8)
        self.assertTrue(dataset.num_attributes == 4)
        self.assertTrue(dataset.inputs.num_attributes == 3)
        self.assertTrue(dataset.targets.num_attributes == 1)

    def test_access(self):
        dataset = Dataset(self.file)

        row = dataset.get(0)
        col = dataset.get_attribute_column(0)

        self.assertTrue(dataset.is_continuous(0))
        self.assertTrue(dataset.is_continuous(1))
        self.assertFalse(dataset.is_continuous(2))
        self.assertFalse(dataset.is_continuous(3))

        self.assertTrue(row.shape[0] == 4)
        self.assertTrue(col.shape[0] == 8)

        self.assertTrue(dataset.attribute_name(3) == "class")
        self.assertTrue(dataset.attribute_name(0) == 'x1')

    def test_attributes(self):
        dataset = Dataset(self.file)

        datum = dataset.get(0)
        self.assertTrue(datum[0] == 0.1)
        self.assertTrue(datum[1] == 0.1)
        self.assertTrue(datum[2] == 0)
        self.assertTrue(datum[3] == 0)

    def test_splits(self):
        dataset = Dataset(self.file)
        set1, set2 = dataset.split()
        self.assertTrue(set1.size == 4)
        self.assertTrue(set2.size == 4)

        dataset = Dataset(self.file)
        set1, set2, set3 = dataset.split([1/3, 1/3, 1/3])
        self.assertTrue(set1.size == 2)
        self.assertTrue(set2.size == 2)
        self.assertTrue(set3.size == 4)

    def test_norm(self):
        #TODO test
        pass

    def test_shuffle(self):
        #TODO test
        pass
