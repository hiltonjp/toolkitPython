import random
import numpy as np
import re


class Dataset(object):
    """Dataset(self, arff=None)


    """

    # STATIC variables
    MISSING = float("infinity")

    def __init__(self, arff=None):
        """
        If matrix is provided, all parameters must be provided, and the new matrix will be
        initialized with the specified portion of the provided matrix.
        """
        self._data = []
        self._attributes = []
        self._str_to_enum = []
        self._enum_to_str = []
        self._dataset_name = "Untitled"

        # iterator helpers
        self._cur = 0

        if arff:
            self._load_arff(arff)

    @property
    def name(self):
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
        d = Dataset()
        d._data = self._data[:, :-1]
        d._attributes = self._attributes
        d._str_to_enum = self._str_to_enum
        d._enum_to_str = self._enum_to_str
        d._dataset_name = self._dataset_name
        return d

    @property
    def targets(self):
        """Expected target values."""

        d = Dataset()
        d._data = self._data[:, -1]
        d._data = d._data[:, np.newaxis]

        d._attributes = self._attributes
        d._str_to_enum = self._str_to_enum
        d._enum_to_str = self._enum_to_str
        d._dataset_name = self._dataset_name
        return d


    def get(self, idx):
        return self._data[idx]

    def get_attribute_column(self, idx):
        return self._data[:, idx]

    def attribute_name(self, col):
        return self._attributes[col]

    def attribute_value(self,col,val):
        return self._enum_to_str[col][val]

    def is_continuous(self, col=0):
        num = len(self._enum_to_str[col]) if len(self._enum_to_str) > 0 else 0
        return True if num == 0 else False

    def split(self, splits=[0.5, 0.5]):
        datasets = []

        for i, split in enumerate(splits):
            if i == len(splits)-1:
                datasets.append(self)
            else:
                d = Dataset()
                d._dataset_name = self._dataset_name
                d._attributes = self._attributes
                d._str_to_enum = self._str_to_enum
                d._enum_to_str = self._enum_to_str

                num_entries = int(split*self.size)
                d._data = self._data[:num_entries]
                self._data = self._data[num_entries:]

                datasets.append(d)

        return datasets

    def shuffle(self, buddy=None):
        """Shuffle the dataset.

        If a buddy dataset is provided, it will be shuffled in the same order.
        """

        if not buddy:
            random.shuffle(self._data)
        else:
            c = list(zip(self._data, buddy.data))
            random.shuffle(c)
            self._data, buddy.data = zip(*c)

    def normalize(self):
        """Normalize each continuous attribute."""
        for i in range(self.num_attributes):
            if self.is_continuous(i):
                col = self.get_attribute_column(i)
                min = np.min(col)
                max = np.max(col)
                col = (col-min)/(max-min)
                self._data[:, i] = col

    def numpy(self):
        return self._data

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

    def __getitem__(self, item):
        """Index operator override. Enables for-each iteration over the class."""
        return self.get(item)

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
