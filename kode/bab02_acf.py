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
    d = y - y.mean()

    print("(1) simpangan dan hasil kali lag h:")
    print("    d =", d.astype(int))
    for h in (1, 2, 3):
        hk = (d[:T - h] * d[h:]).astype(int)
        print(f"    h = {h}: {hk}  jumlah {hk.sum():3d}")

    c = autokov(y, 7)
    r = acf_sampel(y, 7)
    print("(2) autokovarians dan ACF sampel:")
    print("    c(h) =", np.round(c[:4], 4))
    print("    r(h) =", np.round(r[:4], 4))
    print("    statsmodels acf:", np.round(acf(y, nlags=3), 4))
    print(f"    jumlah r(1..7) = {r[1:].sum():.4f}")

    G = np.array([[c[abs(i - j)] for j in range(3)] for i in range(3)])
    print("(3) matriks autokovarians 3 x 3:")
    for baris in G:
        print("    [" + " ".join(f"{v:6.3f}".rstrip("0").ljust(6)
                              for v in baris).rstrip() + "]")
    print(f"    det = {np.linalg.det(G):.4f},"
          f" nilai eigen = {np.round(np.linalg.eigvalsh(G), 4)}")

    b = 1.96 / np.sqrt(T)
    vb = (1 + 2 * (r[1] ** 2 + r[2] ** 2)) / T
    print("(4) batas white noise dan Bartlett (h = 3):")
    print(f"    1.96/sqrt(T) = {b:.4f}")
    print(f"    Var r(3) ~ {vb:.4f}, batas Bartlett = {1.96 * np.sqrt(vb):.4f}")
    print(f"    |r(3)| = {abs(r[3]):.4f}")

    Q, p = ljung_box(y, 2)
    lb = acorr_ljungbox(y, lags=[2], boxpierce=True)
    print("(5) Ljung-Box dan Box-Pierce, H = 2:")
    print(f"    Q = {Q:.4f}, nilai-p = {p:.4f}, exp(-Q/2) = {np.exp(-Q / 2):.4f}")
    print(f"    Box-Pierce = {T * (r[1]**2 + r[2]**2):.4f}")
    print(f"    statsmodels: Q = {lb['lb_stat'].iloc[0]:.4f},"
          f" p = {lb['lb_pvalue'].iloc[0]:.4f},"
          f" BP = {lb['bp_stat'].iloc[0]:.4f}")
