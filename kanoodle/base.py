import os

import numpy as np


TABLE_DIR = os.path.join(__file__.split("base.py")[0], "tables")


class Kanoodle(object):
    def __init__(self, input_path, nrow):
        # Input Path
        self.input_path = input_path

        # Data/Parse
        data = open(self.input_path, "r").read()
        data = np.array([row.split("\t") for row in data.split("\n")])

        # Get Blocks
        is_empty = data == ""
        ix = np.where(is_empty.all(axis=1))[0]
        ix = np.insert(ix, 0, -1)
        self.blocks = [data[(i + 1) : j] for i, j in zip(ix[:-1], ix[1:])]

        # Get Shape
        self.nrow = nrow
        self.ncol = (~is_empty).sum() / self.nrow
        assert int(self.ncol) == self.ncol
        self.ncol = int(self.ncol)

        # Row and Column Indices
        self.row_indices = np.expand_dims(np.arange(self.nrow), (0, 2))
        self.col_indices = np.expand_dims(np.arange(self.ncol), (0, 1))
