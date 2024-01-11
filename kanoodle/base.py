import itertools
import os

import cvxpy
import numpy as np
import pandas as pd


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
        #     2,                         # flip
        #     4,                         # rotation
        #     self.row_indices.shape[1], # row position
        #     self.col_indices.shape[2], # column position
        # ])
        self.nxs = np.prod(
            [  # one x per combination of
                len(self.blocks),  # block
                2,  # flips
                4,  # rotation
                len(self.row_indices),  # row position
                len(self.col_indices),  # column position
            ]
        )
        self.nxs_per_block = self.nxs // len(
            self.blocks
        )  # per block, x(s) are adjacent
        self.nys = self.nrow * self.ncol  # one y per filled position
        self.nvars = self.nxs + self.nys

        # Problem Data
        self.problem_data = None

    def setup_problem(self, force_refresh=False):
        if (self.problem_data is None) or force_refresh:
            # Compute A01, and A02 in A01@x=y and A02@x=1
            col_ix = -1
            x_equals_0 = []  # can set these to 0 to save optimization variables
            x_to_move = dict()  # map i from x_i to action

            A01_x_to_y = np.zeros((self.nys, self.nxs))
            A02_x_eq_1 = np.zeros((len(self.blocks), self.nxs))
            for b, block in enumerate(self.blocks):
                unique_fillings = set()
                for f in [False,True]:
                    flipper = lambda x: x
                    if f:
                        flipper = np.flipud
                    for k in range(4):
                        i, j = np.where(np.rot90(flipper(block), k) != "")
                        rotation = "".join(i.astype(str)) + "," + "".join(j.astype(str))
                        if rotation not in unique_fillings:
                            unique_fillings.add(rotation)
                            # i = np.expand_dims(i,(1,2))
                            # j = np.expand_dims(j,(1,2))
                            # i = i + self.row_indices + self.col_indices
                            # j = j + self.row_indices + self.col_indices
                            # for di, dj in itertools.product(self.row_indices[0,:,0], self.col_indices[0,0,:])
                            for di, dj in itertools.product(
                                self.row_indices, self.col_indices
                            ):
                                col_ix += 1
                                x_to_move[col_ix] = (b, f, k, di, dj)
                                i_ = i + di
                                j_ = j + dj
                                ri_ = i_ * self.ncol + j_
                                valid_indices = (i_ < self.nrow) & (j_ < self.ncol)
                                A01_x_to_y[ri_[valid_indices], col_ix] = 1
                                if (~valid_indices).sum():
                                    x_equals_0.append(col_ix)
                        else:
                            for di, dj in itertools.product(
                                self.row_indices, self.col_indices
                            ):
                                # for di, dj in itertools.product(self.row_indices[0,:,0], self.col_indices[0,0,:])
                                col_ix += 1
                                x_to_move[col_ix] = (b, f, k, di, dj)
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

            # Construct Objective Function Coefficient
            a = np.zeros(self.nvars)
            a[-self.nys :] = 1

            self.problem_data = {
                "a": a,
                "C": C,
                "b": b,
                "x_to_move": x_to_move,
            }

        return self.problem_data

    def solve(self, tol=1e-6, cons=None, force_refresh=True):
        # Problem
        problem_data = self.setup_problem(force_refresh)
        xy_var = cvxpy.Variable((self.nvars, 1), boolean=True)
        objf = cvxpy.Maximize(problem_data["a"] @ xy_var)
        if cons is None:
            cons = []
        else:
            cons = [lhs @ xy_var == rhs for lhs, rhs in cons]
        cons.append(problem_data["C"] @ xy_var == problem_data["b"])

        # Solve
        prob = cvxpy.Problem(objf, cons)
        oofv = prob.solve(solver="ECOS_BB")

        # Index
        idx = list(problem_data["x_to_move"].values()) + [
            ("y", "", "", i, j) for i in range(self.nrow) for j in range(self.ncol)
        ]
        idx = pd.MultiIndex.from_tuples(idx)

        # Return
        if oofv < 0:
            xy_var_val = pd.DataFrame(np.nan, index=idx)[0].rename(None)
            print("Infeasible!")
        else:
            xy_var_val = pd.DataFrame(xy_var.value, index=idx)[0].rename(None)
            xy_var_val = xy_var_val[xy_var_val.abs() > tol]
        return {"problem_data": problem_data, "solution": xy_var_val}

    def show_solution(self, res):
        filled = np.empty((self.nrow, self.ncol), dtype=str)
        for b, f, r, i, j in res["solution"].drop("y").index:
            flipper = lambda x: x
            if f:
                flipper = np.flipud
            block = np.rot90(flipper(self.blocks[b]), r)
            n, m = block.shape
            n_,m_ = filled[slice(i, i + n), slice(j, j + m)].shape
            filled[slice(i, i + n), slice(j, j + m)] = np.core.defchararray.add(
                filled[slice(i, i + n), slice(j, j + m)], block[:n_,:m_]
            )
        return filled
