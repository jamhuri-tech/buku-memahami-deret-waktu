"""Bab 2: autokovarians, ACF sampel, batas white noise, Ljung-Box.

Fungsi di berkas ini dipakai bab-bab berikutnya:
  autokov(y, H)    c(0), ..., c(H) dengan pembagi T
  acf_sampel(y, H) r(0), ..., r(H)
  ljung_box(y, H)  statistik Q dan nilai-p chi-kuadrat(H - k)
"""
import numpy as np
from scipy import stats


def autokov(y, H):
    """Autokovarians sampel c(h) = (1/T) sum (y_t - ybar)(y_{t+h} - ybar)."""
    y = np.asarray(y, dtype=float)
    T = len(y)
    d = y - y.mean()
    return np.array([d[:T - h] @ d[h:] / T for h in range(H + 1)])


def acf_sampel(y, H):
    """Autokorelasi sampel r(h) = c(h) / c(0), h = 0, ..., H."""
    c = autokov(y, H)
    return c / c[0]


def ljung_box(y, H, k=0):
    """Q = T(T+2) sum_{h=1}^{H} r(h)^2 / (T-h), derajat bebas H - k."""
    T = len(y)
    r = acf_sampel(y, H)
    h = np.arange(1, H + 1)
    Q = T * (T + 2) * np.sum(r[1:] ** 2 / (T - h))
    return Q, stats.chi2.sf(Q, H - k)


if __name__ == "__main__":
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf

    from bab01_data import deret_mini

    np.set_printoptions(suppress=True)
    y = deret_mini()
    T = len(y)
    c = autokov(y, 7)
    r = acf_sampel(y, 7)
    G = np.array([[c[abs(i - j)] for j in range(3)] for i in range(3)])
    print("(1) ACF sampel data mini, NumPy dan statsmodels:")
    print("    r(0..3) NumPy      :", np.round(r[:4], 4))
    print("    r(0..3) statsmodels:", np.round(acf(y, nlags=3), 4))
    print(f"    jumlah r(1..7) = {r[1:].sum():.4f}")
    print(f"    det Gamma_3 = {np.linalg.det(G):.4f},"
          f" nilai eigen = {np.round(np.linalg.eigvalsh(G), 4)}")

    Q, p = ljung_box(y, 2)
    lb = acorr_ljungbox(y, lags=[2], boxpierce=True)
    print("(2) Ljung-Box dan Box-Pierce, H = 2:")
    print(f"    NumPy      : Q = {Q:.4f}, p = {p:.4f},"
          f" BP = {T * (r[1]**2 + r[2]**2):.4f}")
    print(f"    statsmodels: Q = {lb['lb_stat'].iloc[0]:.4f},"
          f" p = {lb['lb_pvalue'].iloc[0]:.4f},"
          f" BP = {lb['bp_stat'].iloc[0]:.4f}")
