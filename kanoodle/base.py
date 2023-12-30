import numpy as np


class Kanoodle(object):

    def __init__(self, input_path="./input.txt", nrow=3):

        # Data/Parse
        data = open("./input.txt", "r").read()
        data = np.array([row.split("\t") for row in data.split("\n")])

        # Get Blocks
        is_empty = data==""
        ix = np.where(is_empty.all(axis=1))[0]
        ix = np.insert(ix,0,-1)
        self.blocks = [data[(i+1):j] for i,j in zip(ix[:-1], ix[1:])]

        # Get Shape
        self.nrow = nrow
        self.ncol = (~is_empty).sum() / self.nrow
        assert int(self.ncol) == self.ncol
        self.ncol = int(self.ncol)
