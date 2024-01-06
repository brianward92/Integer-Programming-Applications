import itertools
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
        self.blocks.append(data[(ix[-1] + 1) :])

        # Get Shape
        self.nrow = nrow
        self.ncol = (~is_empty).sum() / self.nrow
        assert int(self.ncol) == self.ncol
        self.ncol = int(self.ncol)

        # Row and Column Indices
        # self.row_indices = np.expand_dims(np.arange(self.nrow), (0, 2))
        # self.col_indices = np.expand_dims(np.arange(self.ncol), (0, 1))
        self.row_indices = np.arange(self.nrow)
        self.col_indices = np.arange(self.ncol)

        # Variable Counts
        # self.nxs = np.prod([           # one x per combination of
        #     len(self.blocks),          # block
        #     4,                         # rotation
        #     self.row_indices.shape[1], # row position
        #     self.col_indices.shape[2], # column position
        # ])
        self.nxs = np.prod(
            [  # one x per combination of
                len(self.blocks),  # block
                4,  # rotation
                len(self.row_indices),  # row position
                len(self.col_indices),  # column position
            ]
        )
        self.nxs_per_block = self.nxs // len(
            self.blocks
        )  # per block, x(s) are adjacent
        self.nys = self.nrow * self.ncol  # one y per filled position

    def setup_problem(self):
        # Compute A01, and A02 in A01@x=y and A02@x=1
        col_ix = -1
        x_equals_0 = []  # can set these to 0 to save optimization variables
        x_to_move = dict()  # map i from x_i to action

        A01_x_to_y = np.zeros((self.nys, self.nxs))
        A02_x_eq_1 = np.zeros((len(self.blocks), self.nxs))
        for b, block in enumerate(self.blocks):
            unique_rotations = set()
            for k in range(4):
                i, j = np.where(np.rot90(block, k) != "")
                rotation = "".join(i.astype(str)) + "," + "".join(j.astype(str))
                if rotation not in unique_rotations:
                    unique_rotations.add(rotation)
                    # i = np.expand_dims(i,(1,2))
                    # j = np.expand_dims(j,(1,2))
                    # i = i + self.row_indices + self.col_indices
                    # j = j + self.row_indices + self.col_indices
                    # for di, dj in itertools.product(self.row_indices[0,:,0], self.col_indices[0,0,:])
                    for di, dj in itertools.product(self.row_indices, self.col_indices):
                        col_ix += 1
                        x_to_move[col_ix] = (b, k, di, dj)
                        i_ = i + di
                        j_ = j + dj
                        ri_ = i_ * self.ncol + j_
                        valid_indices = (i_ < self.nrow) & (j_ < self.ncol)
                        A01_x_to_y[ri_[valid_indices], col_ix] = 1
                        if (~valid_indices).sum():
                            x_equals_0.append(col_ix)
                else:
                    for di, dj in itertools.product(self.row_indices, self.col_indices):
                        # for di, dj in itertools.product(self.row_indices[0,:,0], self.col_indices[0,0,:])
                        col_ix += 1
                        x_to_move[col_ix] = (b, k, di, dj)
                        x_equals_0.append(col_ix)
            s = slice(b * self.nxs_per_block, (b + 1) * self.nxs_per_block)
            A02_x_eq_1[b, s] = 1

        # Compute A03 in A03@x=0 (for x variables that can be pre-opt'd)
        A03_x_eq_0 = np.zeros((self.nxs, self.nxs))
        A03_x_eq_0[x_equals_0, x_equals_0] = 1
        A03_x_eq_0 = A03_x_eq_0[~(A03_x_eq_0 == 0).all(axis=1), :]

        # Concatenate Constraint Matrix
        C = np.vstack(
            [
                np.hstack(
                    [
                        A01_x_to_y,
                        -np.eye(self.nys),
                    ]
                ),
                np.hstack(
                    [
                        A02_x_eq_1,
                        np.zeros((len(self.blocks), self.nys)),
                    ]
                ),
                np.hstack(
                    [
                        A03_x_eq_0,
                        np.zeros((A03_x_eq_0.shape[0], self.nys)),
                    ]
                ),
                np.hstack(
                    [
                        np.zeros((self.nys, self.nxs)),
                        np.eye(self.nys),
                    ]
                ),
            ]
        )

        # Construct RHS
        b = np.vstack(
            [
                np.zeros((A01_x_to_y.shape[0], 1)),
                np.ones((A02_x_eq_1.shape[0], 1)),
                np.zeros((A03_x_eq_0.shape[0], 1)),
                np.ones((self.nys, 1)),
            ]
        )

        return {"C": C, "b": b, "x_to_move": x_to_move}
