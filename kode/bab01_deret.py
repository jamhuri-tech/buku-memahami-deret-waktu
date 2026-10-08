"""Bab 1: lag, diferensi, tabel lag, kuadrat terkecil, dan ramalan."""
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

from bab01_data import deret_mini, rancang, ramal, tabel_lag

np.set_printoptions(suppress=True)
y = deret_mini()

print("(1) deret, lag, dan diferensi:")
print("    t      :", np.arange(1, 9))
print("    y      :", y.astype(int))
print("    rata-rata =", y.mean())
print("    dif 1  :", np.diff(y).astype(int))
print("    dif 2  :", np.diff(y, 2).astype(int))

t, F, target = tabel_lag(y, 2)
X = rancang(F)
print("(2) tabel lag p = 2:")
print("     t  x1  x2  y")
for i in range(len(t)):
    print(f"    {t[i]:2d}  {F[i, 0]:2.0f}  {F[i, 1]:2.0f}  {target[i]:2.0f}")
print("    X'X =", (X.T @ X).astype(int).tolist())
print("    X'y =", (X.T @ target).astype(int).tolist())
print(f"    det X'X = {np.linalg.det(X.T @ X):.0f}")

print("(3) satu baris (t = 3), w = (0, 0, 0), eta = 0.02:")
w = np.zeros(3)
x3, y3 = X[0], target[0]
e = y3 - x3 @ w
print(f"    ramalan = {x3 @ w:.2f}, galat = {e:.2f}, loss = {0.5 * e**2:.2f}")
print("    turunan -e x =", -e * x3)
w = w + 0.02 * e * x3
print("    bobot baru =", np.round(w, 4))
print(f"    ramalan baru = {x3 @ w:.2f}")

w = np.linalg.solve(X.T @ X, X.T @ target)
yh = X @ w
r = target - yh
print("(4) kuadrat terkecil:")
print("    w =", np.round(w, 4))
print("    ramalan :", yh)
print("    residu  :", r)
print(f"    JKG = {r @ r:.4f}, s2 = JKG/(m - 3) = {r @ r / 3:.4f}")

fit = AutoReg(y, lags=2, trend="c").fit()
print("(5) statsmodels AutoReg(lags=2):")
print("    params =", np.round(fit.params, 4))
print(f"    sigma2 = {fit.sigma2:.4f}  (JKG/m)")

f = ramal(y, w, 6)
print("(6) ramalan rekursif t = 9, ..., 14:")
print("    ", " ".join(f"{v:.6f}" for v in f))
print("    statsmodels:")
print("    ", " ".join(f"{v:.6f}" for v in fit.predict(start=8, end=13)))
print(f"    rata-rata jangka panjang w0/(1 - w1 - w2) ="
      f" {w[0] / (1 - w[1] - w[2]):.4f}")
