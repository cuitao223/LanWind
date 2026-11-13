import numpy as np

def wind_direction(u, v):
    return np.degrees(np.arctan2(-u, -v)) % 360

def horizontal_shear(uv, mask, spacing=200.):
    """Central vector-gradient Frobenius norm; shape (..., 2, Z, Y, X).
    Both immediate neighbours and centre must be valid. No zero-filling gradient.
    """
    u, v = uv[..., 0, :, :, :], uv[..., 1, :, :, :]
    valid = mask.copy()
    result = np.full_like(u, np.nan, dtype=np.float32)
    good = (valid[..., 1:-1, 1:-1] & valid[..., :-2, 1:-1] &
            valid[..., 2:, 1:-1] & valid[..., 1:-1, :-2] & valid[..., 1:-1, 2:])
    derivatives = [(a[..., 2:, 1:-1]-a[..., :-2, 1:-1])/(2*spacing) for a in (u,v)]
    derivatives += [(a[..., 1:-1, 2:]-a[..., 1:-1, :-2])/(2*spacing) for a in (u,v)]
    value = np.sqrt(sum(d*d for d in derivatives))
    result[..., 1:-1, 1:-1] = np.where(good, value, np.nan)
    return result
