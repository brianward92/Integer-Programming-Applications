import os

from games.kanoodle import base


def get_simple_example():
    k = base.Kanoodle(
        input_path=os.path.join(base.TABLE_DIR, "input_simple.txt"),
        nrow=3,
    )
    return k
