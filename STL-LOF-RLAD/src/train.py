"""Training entry points.

For submission reproducibility, scripts/run_main_experiment.py records the
locked result JSON rather than changing model logic or retuning parameters.
"""

from .utils import set_seed


def initialize_reproduction(seed: int = 42) -> None:
    set_seed(seed)
