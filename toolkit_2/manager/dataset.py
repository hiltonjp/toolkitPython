import random
import numpy as np
import re
from copy import copy


class Dataset(object):
    """Dataset(self, arff=None)

    An Attribute-Relation Dataset.

    NOTE: This class makes two assumptions:
        1. That the target attribute is given as the LAST attribute in the file.
        2. Nominal (discrete) attributes are treated as a single column in the
           underlying numpy array, and so are really floats ranging from 0-N,
           with N being the number of enumerated values for that given
           attribute. Names for these

    Args:
        arff (str): The path to an ARFF file.

    Attributes:
        name (str): The name of the dataset.

        size (int): The number of instances (data entries) in the dataset.

        num_attributes (int): The number of attributes each instance has,
            including the target attribute.

        inputs (numpy.ndarray): The input attributes to be used in training.
            If you have M instances of data in your dataset, and N non-target
            attributes, then Dataset.inputs will yield a numpy array for shape
            (M, N), that is, M rows and N columns.

        targets (numpy.ndarray): The target attributes to be used in training.
            If you have M instances, then Dataset.targets yields a numpy array
            of shape (M, 1), that is, M rows and 1 column.
    """

    # STATIC variables
    MISSING = float("infinity")

    def __init__(self, arff=None):

        # DO NOT ACCESS THESE DIRECTLY
        self._data = None
        self._attributes = None
        self._str_to_enum = None
        self._enum_to_str = None
        self._dataset_name = None

        if arff:
            self._load_arff(arff)

    ############################################################################
    # ATTRIBUTES                                                               #
    ############################################################################

    @property
    def name(self):
        """The name of the dataset."""
        return self._dataset_name

    @property
    def size(self):
        """The number of data entries in the dataset."""
        return self._data.shape[0]

    @property
    def num_attributes(self):
        """The number of data attributes in the dataset."""
        return self._data.shape[1]

    @property
    def inputs(self):
        """The input features of the data."""
        return self._data[:, :-1]

    @property
    def targets(self):
        """Expected target values."""
        return self._data[:, -1]

    ############################################################################
    # PUBLIC API                                                               #
    ############################################################################

    def get_data(self):
        return self._data

    def set_data(self, data):
        self._data = data

    def numpy(self):
        """Get the numpy array containing all data for the set.

        This includes both input attributes and respective targets.

        Returns:
            (numpy.ndarray): A numpy array containing your data for the set.
            If you have M instances of data, and N attributes defined
            (including the target), then the returned array will have a
            shape of (M, N)--M rows, N columns.
        """
        return self._data

    def attribute_name(self, index):
        """Gets the name of an attribute defined in the ARFF file.

        Attribute names are stored in the order presented in the ARFF file.

        Args:
            index (int): The index of the desired attribute.

        Returns:
            (str): The name of the attribute.
        """
        return self._attributes[index]

    def nominal_labels(self, attr_index):
        """Get the labels for a nominal attribute's enumerated values.

        Args:
            attr_index (int): The index of the desired attribute

        Returns:
            (list[str...]): A list of the labels given to that attribute's
                enumerated values.
        """
        try:
            return self._enum_to_str[attr_index]
        except IndexError:
            raise IndexError(f"Attribute {attr_index} is not nominal.")

    def is_continuous(self, attr_index):
        """Check if an attribute from the ARFF file is continuous.

        Args:
            attr_index (int): The index of the desired attribute

        Returns:
            (bool): True if the attribute is continuous; False otherwise.
        """

        num = len(self._enum_to_str[attr_index]) if len(self._enum_to_str) > 0 else 0
        return True if num == 0 else False

    def is_nominal(self, attr_index):
        """Check if an attribute from the ARFF file is nominal.

        Args:
            attr_index (int): The index of the desired attribute

        Returns:
            (bool): True if the attribute is nominal; False otherwise.
        """
        return not self.is_continuous(attr_index)

    def split(self, split=(0.5, 0.5)):
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
        if type(split) == float:
            d = Dataset()
            d._dataset_name = self._dataset_name
            d._attributes = self._attributes
            d._str_to_enum = self._str_to_enum
            d._enum_to_str = self._enum_to_str

            num_entries = int(split * self.size)
            d._data = self._data[:num_entries]
            self._data = self._data[num_entries:]

            return d, self
        elif len(split) == 1:
            return [self]
        else:
            first, rest = self.split(split[0])
            return first, *rest.split(split[1:])

    def shuffle(self):
        """Shuffle the dataset randomly."""
        np.random.shuffle(self._data)

    def normalize(self):
        """Normalize each continuous attribute.

        TODO: Is it a bad thing to only normalize continuous data?
        """
        for i in range(self.num_attributes):
            if self.is_continuous(i):
                col = self._data[:, i]
                min = np.min(col)
                max = np.max(col)
                col = (col-min)/(max-min)
                self._data[:, i] = col

    ############################################################################
    # PRIVATE FUNCTIONS                                                        #
    ############################################################################

    def _load_arff(self, filename):
        self._data = []
        self._attributes.clear()
        self._str_to_enum.clear()
        self._enum_to_str.clear()
        reading_data = False

        data = []
        f = open(filename, "r")
        for line in f.readlines():
            line = line.rstrip()

            # Skip comments and spacing
            if len(line) == 0 or line[0] == '%':
                continue

            # Parse everything up to the "@DATA" header
            if not reading_data:

                lower = line.lower()  # Just to save time on branches
                if lower.startswith("@relation"):
                    self._load_relation(line)

                elif lower.startswith("@attribute"):
                    self._load_attribute(line)

                elif lower.startswith("@data"):
                    reading_data = True

            # Parse all of the data and load into an array.
            else:
                datapoint = self._load_datapoint(line)
                data.append(datapoint)

        self._data = np.array(data)

    def _load_relation(self, arff_line):
        name = arff_line[9:].strip()
        assert " " not in name, "Dataset name may not contain spaces."
        self._dataset_name = name

    def _load_attribute(self, arff_line):
        """clean up and store attribute name and values from an ARFF line."""

        arff_line = arff_line[10:].strip()

        # Get attribute name and definition
        if arff_line[0] == "'":
            arff_line = arff_line[1:]

            attr_name = arff_line[:arff_line.index("'")]
            attr_def = arff_line[arff_line.index("'")+1:].strip()
        else:
            search = re.search(r'(\w*)\s*(.*)', arff_line)
            attr_name = search.group(1)
            attr_def = search.group(2)

            # Remove white space from attribute values
            attr_def = "".join(attr_def.split())

        self._attributes.append(attr_name)

        # Parse discrete values, if any.
        str_to_enum = {}
        enum_to_str = {}

        if not (attr_def.lower() == "real" or attr_def.lower() == "continuous" or attr_def.lower() == "integer"):
            assert attr_def[0] == '{' and attr_def[-1] == '}', \
                "Enclose discrete values with {,}."

            attr_def = attr_def[1:-1]
            attr_vals = attr_def.split(",")

            for idx, val in enumerate(attr_vals):
                val = val.strip()
                enum_to_str[idx] = val
                str_to_enum[val] = idx

        self._enum_to_str.append(enum_to_str)
        self._str_to_enum.append(str_to_enum)

    def _load_datapoint(self, arff_line):
        """Load a single point of data from the ARFF file."""

        row = []
        for idx, val in enumerate(arff_line.split(",")):
            val = val.strip()
            if not val:
                pass
            else:
                # Tries to convert val into continuous value.
                # If this fails, then the value is discrete and float() casts
                # the string into an enumeration.
                datum = float(self.MISSING if val == '?' else self._str_to_enum[idx].get(val, val))
                row.append(datum)

        return row

    ############################################################################
    # OPERATOR OVERRIDES                                                       #
    ############################################################################

    def __copy__(self):
        d = Dataset()
        d._dataset_name = self._dataset_name
        d._attributes = self._attributes
        d._str_to_enum = self._str_to_enum
        d._enum_to_str = self._enum_to_str
        d._data = self._data

        return d

    def __add__(self, other):
        """Merge two datasets together."""
        d = copy(self)
        d._data = np.concatenate([self._data, other._data], axis=0)
        return d

    def __getitem__(self, index):
        """Index operator override. Enables for-each iteration over the class."""
        return self._data[index]

    def __len__(self):
        """Number of data entries."""
        return self.size

    def __str__(self):
        """Returns the ARFF file format of this data."""
        s = f"@RELATION {self._dataset_name}\n"
        for i in range(len(self._attributes)):
            s += f"@ATTRIBUTE {self._attributes[i]} "
            if self.is_continuous(i):
                s += "CONTINUOUS\n"
            else:
                s += "{" + ", ".join(self._enum_to_str[i].values()) + "}"

        s += "@DATA"
        for i in range(self.size):
            row = self.get(i)
            values = [
                str(row[j]) if self.is_continuous(i) \
                else self._enum_to_str[j][row[j]] \
                for j in range(len(row))
            ]

            s += ", ".join(values)

        return s
