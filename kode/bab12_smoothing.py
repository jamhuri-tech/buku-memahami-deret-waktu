"""Bab 12: exponential smoothing sederhana, Holt-Winters aditif.

Fungsi yang dipakai bab-bab berikutnya:
  ses(y, alpha, l0)                    tingkat l_t dan galat e_t
  holt_winters(y, a, b, g, l0, b0, s0) tingkat, tren, musiman, galat
"""
import warnings

import numpy as np


def ses(y, alpha, l0):
    """l_t = l_{t-1} + alpha e_t, e_t = y_t - l_{t-1}."""
    l, ls, es = l0, [], []
    for v in y:
        e = v - l
        l = l + alpha * e
        ls.append(l)
        es.append(e)
    return np.array(ls), np.array(es)


def holt_winters(y, a, b, g, l0, b0, s0):
    """Holt-Winters aditif; s0 = musiman untuk t = 1-m, ..., 0.

    e_t = y_t - (l + tr + s_{t-m});
    l_t = a (y_t - s_{t-m}) + (1 - a)(l_{t-1} + b_{t-1});
    b_t = b (l_t - l_{t-1}) + (1 - b) b_{t-1};
    s_t = g (y_t - l_{t-1} - b_{t-1}) + (1 - g) s_{t-m}.
    """
    l, tr, s = l0, b0, list(s0)
    L, Bt, E = [], [], []
    for t, v in enumerate(y):
        sm = s[t]
        e = v - (l + tr + sm)
        l_baru = a * (v - sm) + (1 - a) * (l + tr)
        s.append(g * (v - l - tr) + (1 - g) * sm)
        tr = b * (l_baru - l) + (1 - b) * tr
        l = l_baru
        L.append(l)
        Bt.append(tr)
        E.append(e)
    return np.array(L), np.array(Bt), np.array(s), np.array(E)


if __name__ == "__main__":
    from scipy.optimize import minimize_scalar
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.holtwinters import (ExponentialSmoothing,
                                             SimpleExpSmoothing)

    from bab01_data import deret_mini, deret_musiman
    from bab07_estimasi import galat_ma1

    warnings.filterwarnings("ignore")
    y = deret_mini()

    l, e = ses(y, 0.5, 2.0)
    sm = SimpleExpSmoothing(y, initialization_method="known",
                            initial_level=2.0).fit(smoothing_level=0.5,
                                                   optimized=False)
    print("(1) SES data mini, alpha = 0.5, l_0 = 2:")
    print("    l_t   :", " ".join(f"{v:.4f}" for v in l[:4]))
    print("           ", " ".join(f"{v:.4f}" for v in l[4:]))
    print(f"    JKG = {e @ e:.4f}, ramalan = {l[-1]:.4f},"
          f" statsmodels = {sm.forecast(1)[0]:.4f}")
    jkg = lambda a: ses(y, a, 2.0)[1] @ ses(y, a, 2.0)[1]
    r = minimize_scalar(jkg, bounds=(0.01, 1), method="bounded")
    print(f"    alpha terbaik (l_0 = 2): {r.x:.3f}, JKG = {r.fun:.4f}")

    d = np.diff(y)
    e_arima, _ = galat_ma1(np.r_[y[0] - 2.0, d], -0.5)
    print("(2) ARIMA(0,1,1) theta = -0.5, galat sama dengan SES:",
          np.allclose(e_arima, e))

    z = deret_musiman()
    L, Bt, S, E = holt_winters(z, 0.5, 0.5, 0.5, 4.0, 1.0, [-3, 1, 4, -2])
    hw = ExponentialSmoothing(z, trend="add", seasonal="add",
                              seasonal_periods=4,
                              initialization_method="known",
                              initial_level=4.0, initial_trend=1.0,
                              initial_seasonal=[-3, 1, 4, -2]).fit(
        smoothing_level=0.5, smoothing_trend=0.5, smoothing_seasonal=0.5,
        optimized=False)
    f = L[-1] + Bt[-1] * np.arange(1, 5) + S[-4:]
    ets = ETSModel(z, error="add", trend="add", seasonal="add",
                   seasonal_periods=4, initialization_method="known",
                   initial_level=4.0, initial_trend=1.0,
                   initial_seasonal=[-3, 1, 4, -2]).smooth([0.5, 0.25, 0.5])
    print("(3) Holt-Winters aditif, alpha = beta = gamma = 0.5:")
    print("    galat t = 1..4:", " ".join(f"{v:.3f}" for v in E[:4]))
    print("    ramalan t = 13..16:")
    print("    NumPy                :", " ".join(f"{v:.3f}" for v in f))
    print("    ETSModel             :", " ".join(
        f"{v:.3f}" for v in ets.forecast(4)))
    print("    ExponentialSmoothing :", " ".join(
        f"{v:.3f}" for v in hw.forecast(4)))
