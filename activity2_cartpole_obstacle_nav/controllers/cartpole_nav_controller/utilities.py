import math

import numpy as np


def normalize_to_range(value, min_val, max_val, new_min, new_max, clip=False):
    """Linearly map value from [min_val, max_val] to [new_min, new_max]."""
    value = (new_max - new_min) / (max_val - min_val) * (float(value) - max_val) + new_max
    return float(np.clip(value, new_min, new_max)) if clip else value


def wrap_angle(angle):
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi
