import os

from ip_problems.kanoodle import base


def get_simple_example():
    k = base.Kanoodle(
        input_path=os.path.join(base.TABLE_DIR, "input_simple.txt"),
        nrow=4,
    )
    return k


def get_actual_kanoodle():
    k = base.Kanoodle(
        input_path=os.path.join(base.TABLE_DIR, "actual_kanoodle.txt"),
        nrow=5,
    )
    return k
