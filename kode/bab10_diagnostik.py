"""Bab 10: kriteria informasi pada sampel yang sama, diagnostik residu.

Fungsi yang dipakai bab-bab berikutnya:
  kriteria(jkg, m, k)  (AIC, AICc, BIC) dengan -2 log L = m log(JKG/m)
"""
import numpy as np


def kriteria(jkg, m, k):
    """AIC, AICc, BIC tanpa konstanta bersama: m log(JKG/m) + penalti."""
    dasar = m * np.log(jkg / m)
    aic = dasar + 2 * k
    return aic, aic + 2 * k * (k + 1) / (m - k - 1), dasar + k * np.log(m)


if __name__ == "__main__":
    from scipy import stats
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.stattools import jarque_bera

    from bab01_data import deret_mini, rancang, tabel_lag

    y = deret_mini()
    t, F, target = tabel_lag(y, 2)            # sampel bersama t = 3..8
    m = len(target)
    print("(1) AR(p) pada sampel yang sama t = 3..8 (m = 6):")
    print("    p   k     JKG     AIC    AICc     BIC")
    for p in range(3):
        X = rancang(F[:, :p]) if p else np.ones((m, 1))
        w = np.linalg.lstsq(X, target, rcond=None)[0]
        e = target - X @ w
        a, ac, b = kriteria(e @ e, m, p + 1)
        print(f"    {p}   {p + 1}  {e @ e:6.4f}  {a:6.3f}  {ac:6.3f}"
              f"  {b:6.3f}")

    X = rancang(F)
    e = target - X @ np.linalg.solve(X.T @ X, X.T @ target)
    lb = acorr_ljungbox(e, lags=[3], model_df=2)
    jb = jarque_bera(e)
    print("(2) diagnostik residu AR(2) data mini:")
    print("    residu =", e)
    print(f"    Ljung-Box H = 3, db = 1: Q = {lb['lb_stat'].iloc[0]:.4f},"
          f" p = {lb['lb_pvalue'].iloc[0]:.4f}")
    print(f"    Jarque-Bera: JB = {jb[0]:.4f}, p = {jb[1]:.4f},"
          f" kurtosis = {jb[3]:.4f}")
