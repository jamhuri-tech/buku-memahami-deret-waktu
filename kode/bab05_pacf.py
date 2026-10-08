"""Bab 5: autokorelasi parsial dengan rekursi Durbin-Levinson.

Fungsi yang dipakai bab-bab berikutnya:
  durbin_levinson(rho, H)  PACF alpha(1..H), bobot phi_{H,j}, dan
                           rasio varians galat ramalan v_h / gamma(0)
  pacf_sampel(y, H)        PACF sampel dari ACF sampel berpembagi T
"""
import numpy as np


def durbin_levinson(rho, H):
    """Rekursi Durbin-Levinson pada autokorelasi rho(0..H)."""
    rho = np.asarray(rho, dtype=float)
    alpha = np.zeros(H)
    v = np.ones(H + 1)
    phi = np.zeros(0)
    for h in range(1, H + 1):
        num = rho[h] - phi @ rho[h - 1:0:-1] if h > 1 else rho[1]
        den = 1 - phi @ rho[1:h] if h > 1 else 1.0
        k = num / den
        phi = np.r_[phi - k * phi[::-1], k]
        alpha[h - 1] = k
        v[h] = v[h - 1] * (1 - k * k)
    return alpha, phi, v


def pacf_sampel(y, H):
    """PACF sampel alpha(1..H) dari r(h) dengan pembagi T."""
    from bab02_acf import acf_sampel
    return durbin_levinson(acf_sampel(y, H), H)[0]


if __name__ == "__main__":
    from statsmodels.tsa.arima_process import ArmaProcess
    from statsmodels.tsa.stattools import pacf

    from bab01_data import deret_mini
    from bab02_acf import acf_sampel, autokov

    np.set_printoptions(suppress=True)
    y = deret_mini()
    a, phi, v = durbin_levinson(acf_sampel(y, 3), 3)
    print("(1) PACF sampel data mini, h = 1, 2, 3:")
    print("    Durbin-Levinson     :", np.round(a, 4))
    print("    statsmodels 'ywm'   :", np.round(pacf(y, 3, method="ywm")[1:], 4))
    print("    statsmodels 'ols'   :", np.round(pacf(y, 3, method="ols")[1:], 4))
    print("    v_h, h = 0..3:", np.round(autokov(y, 0)[0] * v, 4))

    print("(2) ACF dan PACF teoretis, h = 1..5:")
    for nama, ar, ma in [("AR(2) (0.5, -0.5)", [1, -0.5, 0.5], [1]),
                         ("MA(1) theta = 0.5", [1], [1, 0.5])]:
        p = ArmaProcess(ar, ma)
        print(f"    {nama}")
        print("      ACF :", np.round(p.acf(6)[1:], 4) + 0.0)
        print("      PACF:", np.round(p.pacf(6)[1:], 4) + 0.0)
