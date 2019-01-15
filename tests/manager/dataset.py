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
        self.assertTrue(col.shape[1] == 3)

        self.assertTrue(dataset.attribute_name(4) == "class")
        self.assertTrue(dataset.attribute_name(0) == 'x1')