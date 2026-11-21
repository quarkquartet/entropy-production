"""Higgs wall width from the moment equation in Appendix A of the paper."""

import numpy as np
from scipy.integrate import quad


def wall_width(dVdphi, v):
    """Return L = v/sqrt(6 I) using the partial potential derivative."""
    def integrand(x):
        if x <= 0.0 or x >= 1.0:
            return 0.0
        left = 0.5 * v * (1.0 - x)
        right = 0.5 * v * (1.0 + x)
        return np.arctanh(x) * (dVdphi(left) - dVdphi(right))

    integral = 0.5 * v * quad(integrand, 0.0, 1.0)[0]
    if integral <= 0.0:
        raise ValueError(f"non-positive wall-width integral I = {integral:.4g}")
    return v / np.sqrt(6.0 * integral)
