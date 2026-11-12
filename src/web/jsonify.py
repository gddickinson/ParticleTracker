"""Convert pipeline output (numpy scalars/arrays, NaN, tuples) to plain JSON values."""
import math

import numpy as np


def jsonable(obj, digits=None):
    """Recursively convert to JSON-safe Python types; NaN/inf become None.

    digits: round floats to this many significant decimals (None keeps them exact).
    """
    if isinstance(obj, dict):
        return {str(k): jsonable(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v, digits) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist(), digits)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if not math.isfinite(f):
            return None
        return round(f, digits) if digits is not None else f
    if isinstance(obj, np.str_):
        return str(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)
