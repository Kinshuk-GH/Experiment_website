import math
from io import BytesIO
import numpy as np
import pandas as pd

def valid_re(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and 1 <= value <= 10

def validate_table(table):
    used = table.dropna(how="all").copy()
    if len(used) < 2:
        raise ValueError("Enter at least two complete rows.")
    data = used.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Every used row must contain two finite numbers.")
    if not data["Re"].between(1, 10).all():
        raise ValueError("Every Reynolds number must be between 1 and 10.")
    if (data["Cd"] < 0).any():
        raise ValueError("Drag coefficients must be non-negative.")
    if data["Re"].duplicated().any():
        raise ValueError("Use a different Reynolds number in each row.")
    return data.sort_values("Re").reset_index(drop=True)

def figure_bytes(fig, kind):
    buffer = BytesIO()
    fig.savefig(buffer, format=kind, dpi=300, bbox_inches="tight", facecolor="white")
    return buffer.getvalue()
