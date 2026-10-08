"""Bab 3: tren, dummy musiman, rata-rata bergerak, dekomposisi."""
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

from bab01_data import deret_musiman

np.set_printoptions(suppress=True)


def f(v):
    """Deret angka dengan lebar tetap, tiga desimal."""
    return " ".join(f"{x:7.3f}" for x in v)


y = deret_musiman()
T, s = len(y), 4
t = np.arange(1, T + 1)
kw = (t - 1) % s + 1

print("(1) data kuartalan (baris = tahun):")
for th in range(3):
    print(f"    tahun {th + 1}: {y[4 * th:4 * th + 4].astype(int)}"
          f"  jumlah {y[4 * th:4 * th + 4].sum():.0f}")

X = np.column_stack([np.ones(T), t])
w = np.linalg.solve(X.T @ X, X.T @ y)
e = y - X @ w
print("(2) tren linear saja:")
print("    X'X =", (X.T @ X).astype(int).tolist(),
      " X'y =", (X.T @ y).astype(int).tolist())
print(f"    w0 = {w[0]:.4f}, w1 = {w[1]:.4f}")
print("    residu rata-rata per kuartal:")
print("    ", np.round([e[kw == q].mean() for q in range(1, 5)], 4))

D = np.column_stack([(kw == q).astype(float) for q in (2, 3, 4)])
X = np.column_stack([np.ones(T), t, D])
w = np.linalg.solve(X.T @ X, X.T @ y)
e = y - X @ w
print("(3) tren + dummy kuartal 2, 3, 4:")
print("    w =", np.round(w, 4))
print("    residu =", np.round(e, 4) + 0.0)
print(f"    JKG = {e @ e:.4f}, s2 = JKG/(12 - 5) = {e @ e / 7:.4f}")
print("    X'e =", np.round(X.T @ e, 10) + 0.0)

bobot = np.array([1, 2, 2, 2, 1]) / 8
m = np.full(T, np.nan)
for i in range(2, T - 2):
    m[i] = bobot @ y[i - 2:i + 3]
print("(4) rata-rata bergerak terpusat 2x4, t = 3..10:")
print("    m_t * 8 =", (8 * m[2:10]).astype(int))
print("    m_t     =", f(m[2:6]))
print("             ", f(m[6:10]))

d = y - m
idx = np.array([np.nanmean(d[kw == q]) for q in range(1, 5)])
print("(5) dekomposisi aditif klasik:")
print("    y - m, t = 3..10:")
print("    ", f(d[2:6]))
print("    ", f(d[6:10]))
print("    indeks musiman =", f(idx))
print("    jumlah indeks =", idx.sum())
print("    y - S, tahun 1-3:")
for th in range(3):
    print("    ", f((y - idx[kw - 1])[4 * th:4 * th + 4]))

dk = seasonal_decompose(y, model="additive", period=4)
print("(6) statsmodels seasonal_decompose:")
print("    tren    =", f(dk.trend[2:6]))
print("             ", f(dk.trend[6:10]))
print("    musiman =", f(dk.seasonal[:4]))

d4 = y[4:] - y[:-4]
print("(7) diferensi musiman y_t - y_{t-4}, t = 5..12:")
print("    ", d4.astype(int), " rata-rata =", d4.mean())
