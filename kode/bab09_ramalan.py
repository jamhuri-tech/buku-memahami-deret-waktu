"""Bab 9: ramalan ARMA, galat baku ramalan dari bobot psi, selang.

Fungsi yang dipakai bab-bab berikutnya:
  galat_baku_ramalan(phi, theta, sigma2, H)  sqrt(sigma2 sum psi_j^2)
"""
import warnings

import numpy as np

from bab06_arma import bobot_psi


def galat_baku_ramalan(phi, theta, sigma2, H):
    """Galat baku ramalan h = 1..H: sqrt(sigma2 * sum_{j<h} psi_j^2)."""
    psi = bobot_psi(phi, theta, H - 1)
    return np.sqrt(sigma2 * np.cumsum(psi**2))


if __name__ == "__main__":
    from statsmodels.tsa.arima.model import ARIMA

    from bab01_data import deret_mini, ramal
    from bab07_estimasi import galat_ma1

    warnings.filterwarnings("ignore")
    y = deret_mini()

    f = ramal(y, np.array([4, 0.5, -0.5]), 4)
    se = galat_baku_ramalan([0.5, -0.5], [], 1.0, 4)
    sm = ARIMA(y, order=(2, 0, 0), trend="c").fit_constrained(
        {"const": 4, "ar.L1": 0.5, "ar.L2": -0.5, "sigma2": 1.0})
    g = sm.get_forecast(4)
    print("(1) AR(2) data mini, sigma2 = 1, h = 1..4:")
    print("    ramalan      :", " ".join(f"{v:7.4f}" for v in f))
    print("    SE kita      :", " ".join(f"{v:7.4f}" for v in se))
    print("    SE statsm.   :", " ".join(f"{v:7.4f}" for v in g.se_mean))
    lo, hi = g.conf_int().T
    print("    batas bawah  :", " ".join(f"{v:7.4f}" for v in lo))
    print("    batas atas   :", " ".join(f"{v:7.4f}" for v in hi))

    print("(2) galat baku ramalan, sigma2 = 1:")
    print("    h            :      1       2       4       8      16")
    ar = galat_baku_ramalan([0.5, -0.5], [], 1.0, 16)
    rw = galat_baku_ramalan([1.0], [], 1.0, 16)
    for nama, v in (("AR(2) mini", ar), ("random walk", rw)):
        print(f"    {nama:13s}:", " ".join(f"{v[h - 1]:7.4f}"
                                          for h in (1, 2, 4, 8, 16)))
    print(f"    sqrt(gamma(0)) = sqrt(1.5) = {np.sqrt(1.5):.4f}")

    e, _ = galat_ma1(y - 4, 0.3)
    print("(3) MA(1) theta = 0.3, mu = 4:")
    print(f"    e_8 = {e[-1]:.4f}, ramalan h = 1: {4 + 0.3 * e[-1]:.4f},"
          f" h >= 2: 4")
    print("    SE (sigma2 = 1):", np.round(galat_baku_ramalan([], [0.3],
                                                            1.0, 3), 4))
