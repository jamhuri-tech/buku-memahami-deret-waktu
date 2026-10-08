"""Bab 1: kuadrat terkecil AR(2) pada tabel lag dan ramalan rekursif."""
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

from bab01_data import deret_mini, rancang, ramal, tabel_lag

np.set_printoptions(suppress=True)
y = deret_mini()
t, F, target = tabel_lag(y, 2)
X = rancang(F)
w = np.linalg.solve(X.T @ X, X.T @ target)
e = target - X @ w
fit = AutoReg(y, lags=2, trend="c").fit()

print("(1) AR(2) pada tabel lag, NumPy dan statsmodels AutoReg:")
print("    NumPy  w =", np.round(w, 4), f" JKG = {e @ e:.4f}")
print("    AutoReg  =", np.round(fit.params, 4),
      f" sigma2 = {fit.sigma2:.4f}")
print(f"    JKG/m = {e @ e / len(e):.4f}, JKG/(m - 3) = {e @ e / 3:.4f}")

f = ramal(y, w, 5)
print("(2) ramalan rekursif t = 9, ..., 13:")
print("    NumPy      :", " ".join(f"{v:.5f}" for v in f))
print("    statsmodels:", " ".join(
    f"{v:.5f}" for v in fit.predict(start=8, end=12)))
print(f"    w0/(1 - w1 - w2) = {w[0] / (1 - w[1] - w[2]):.4f}")
