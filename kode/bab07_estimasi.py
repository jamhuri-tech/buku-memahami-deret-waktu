"""Bab 7: CSS dan Gauss-Newton untuk MA(1), MLE AR(1), galat baku.

Fungsi yang dipakai bab-bab berikutnya:
  galat_ma1(d, theta)        e_t dan turunan g_t = de_t/dtheta
  gauss_newton_ma1(d, th0)   lintasan Gauss-Newton untuk CSS MA(1)
  loglik_ar1(y, phi, mu, s2) log-kemungkinan eksak AR(1) Gauss
"""
import warnings

import numpy as np


def galat_ma1(d, theta):
    """e_t = d_t - theta e_{t-1}, g_t = -e_{t-1} - theta g_{t-1}."""
    e, g = np.zeros(len(d)), np.zeros(len(d))
    e_lalu = g_lalu = 0.0
    for t, v in enumerate(d):
        g[t] = -e_lalu - theta * g_lalu
        e[t] = v - theta * e_lalu
        e_lalu, g_lalu = e[t], g[t]
    return e, g


def gauss_newton_ma1(d, theta0=0.0, iterasi=30):
    """Lintasan (theta_k, S_k) dengan theta <- theta - g'e / g'g."""
    th, jalur = theta0, []
    for _ in range(iterasi):
        e, g = galat_ma1(d, th)
        jalur.append((th, e @ e))
        th = th - (g @ e) / (g @ g)
    return jalur


def loglik_ar1(y, phi, mu, s2):
    """Log-kemungkinan eksak AR(1) dengan inovasi normal."""
    d = np.asarray(y, dtype=float) - mu
    q = (1 - phi**2) * d[0] ** 2 + np.sum((d[1:] - phi * d[:-1]) ** 2)
    T = len(d)
    return -T / 2 * np.log(2 * np.pi * s2) + 0.5 * np.log(1 - phi**2) \
        - q / (2 * s2)


if __name__ == "__main__":
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.arima.model import ARIMA

    from bab01_data import deret_mini, rancang, tabel_lag
    from bab02_acf import acf_sampel

    warnings.filterwarnings("ignore")
    np.set_printoptions(suppress=True)
    y = deret_mini()
    jalur = gauss_newton_ma1(y - 4, 0.0, 30)
    print("(1) CSS MA(1) data mini (mu = 4), Gauss-Newton dari 0:")
    for k in (0, 1, 2, 3, 5, 10, 29):
        print(f"    k = {k:2d}: theta = {jalur[k][0]:.4f},"
              f" S = {jalur[k][1]:.4f}")

    ols = AutoReg(y, 1).fit()
    mle = ARIMA(y, order=(1, 0, 0)).fit()
    print("(2) AR(1) data mini, tiga penaksir phi:")
    print(f"    kuadrat terkecil (CSS) : {ols.params[1]:.4f}")
    print(f"    Yule-Walker r(1)       : {acf_sampel(y, 1)[1]:.4f}")
    print(f"    MLE eksak (ARIMA)      : {mle.params[1]:.3f},"
          f" mu = {mle.params[0]:.3f}")
    print(f"    log-kem. phi = 0.5, mu = 4, s2 = 1:"
          f" {loglik_ar1(y, 0.5, 4, 1):.4f}")

    t, F, target = tabel_lag(y, 2)
    X = rancang(F)
    V = np.linalg.inv(X.T @ X)
    ar2 = AutoReg(y, 2).fit()
    print("(3) galat baku AR(2) data mini:")
    print("    s2 (X'X)^-1, s2 = 1     :",
          np.round(np.sqrt(np.diag(V))[1:], 4))
    print("    AutoReg (s2 = JKG/m)    :", np.round(ar2.bse[1:], 4))
    G = 1.5 * np.array([[1, 1 / 3], [1 / 3, 1]])
    print("    asimtotik sigma2 G^-1/m :",
          np.round(np.sqrt(np.diag(np.linalg.inv(G)) / 6), 4))
