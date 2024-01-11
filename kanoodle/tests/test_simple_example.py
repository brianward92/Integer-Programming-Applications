import numpy as np

from ip_problems.kanoodle.utils import get_simple_example


def check_solution(xy, C, b):
    assert (C @ xy == b).all()


def get_x_from_solution(solution):
    x = np.zeros(self.nxs)
    for move in solution:
        x[move_to_x[move]] = 1
    x = x.reshape(len(x), 1)
    return x


if __name__ == "__main__":
    # Get Problem
    self = get_simple_example()

    # Extract Setup
    d = self.setup_problem()
    C = d["C"]
    b = d["b"]
    move_to_x = {v: k for k, v in d["x_to_move"].items()}

    # Y Vector Fixed
    y = np.ones(self.nys)
    y = y.reshape(len(y), 1)

    # Hand-Crafted Solutions
    solutions = [
        [
            (0, False, 0, 0, 0),
            (1, False, 0, 1, 0),
            (2, False, 0, 3, 0),
        ],  # sol'n 1: just push pieces together
        [
            (0, False, 0, 1, 0),
            (1, False, 0, 2, 0),
            (2, False, 0, 0, 0),
        ],  # sol'n 2: same as 1, but put Y at top
        [
            (0, False, 2, 1, 0),
            (1, False, 2, 0, 0),
            (2, False, 0, 3, 0),
        ],  # sol'n 3: same as 1, but flip R/G block upside down
        [
            (0, False, 2, 2, 0),
            (1, False, 2, 1, 0),
            (2, False, 0, 0, 0),
        ],  # sol'n 4: same as 3, but put Y at top
    ]

    # Check Solutions
    for solution in solutions:
        x = get_x_from_solution(solution)
        xy = np.vstack([x, y])
        check_solution(xy, C, b)
    else:
        print("All tests passed!")
