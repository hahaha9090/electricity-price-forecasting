import numpy as np

def _pava(row):
    """Unweighted Euclidean projection onto the nondecreasing cone."""
    means, counts = [], []
    for value in row:
        means.append(float(value)); counts.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            n = counts[-2] + counts[-1]
            v = (means[-2] * counts[-2] + means[-1] * counts[-1]) / n
            means[-2:] = [v]; counts[-2:] = [n]
    return np.repeat(means, counts)

def anchor_project(proposals, median):
    """L2 isotonic projection with x[...,4] fixed to the original median."""
    z = np.asarray(proposals, dtype=float); m = np.asarray(median, dtype=float)
    if (z.ndim < 1 or z.shape[-1] != 9 or m.shape != z.shape[:-1] or
            not z.size or not np.isfinite(z).all() or not np.isfinite(m).all()):
        raise ValueError('Finite nine-column proposals and aligned median required')
    result = np.empty_like(z)
    for row, target, out in zip(z.reshape(-1, 9), m.ravel(), result.reshape(-1, 9)):
        out[:4] = np.minimum(_pava(row[:4]), target)
        out[4] = target
        out[5:] = np.maximum(_pava(row[5:]), target)
    return result

def median_sweep(proposals, median):
    """Existing median-outward min-max rearrangement, not an MPAC invention.

    Fakoor et al., JMLR 24 (2023), Flexible Model Aggregation for Quantile
    Regression, section 4.2.1 equation (23). Kept as a cheap strong control.
    """
    z = np.array(proposals, dtype=float, copy=True); m = np.asarray(median, dtype=float)
    if z.shape[-1] != 9 or m.shape != z.shape[:-1] or not np.isfinite(z).all() or not np.isfinite(m).all():
        raise ValueError('Finite aligned proposals/median required')
    z[..., 4] = m
    for j in range(3, -1, -1): z[..., j] = np.minimum(z[..., j], z[..., j+1])
    for j in range(5, 9): z[..., j] = np.maximum(z[..., j], z[..., j-1])
    return z
