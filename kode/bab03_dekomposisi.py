"""Bab 3: regresi tren dan dekomposisi klasik (NumPy dan statsmodels)."""
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

X = np.column_stack([np.ones(T), t])
w = np.linalg.solve(X.T @ X, X.T @ y)
e = y - X @ w
print("(1) tren linear saja:")
print("    X'X =", (X.T @ X).astype(int).tolist(),
      " X'y =", (X.T @ y).astype(int).tolist())
print(f"    w0 = {w[0]:.4f}, w1 = {w[1]:.4f}")
print("    rata-rata residu per kuartal:")
print("    ", f([e[kw == q].mean() for q in range(1, 5)]))

bobot = np.array([1, 2, 2, 2, 1]) / 8
m = np.full(T, np.nan)
for i in range(2, T - 2):
    m[i] = bobot @ y[i - 2:i + 3]
idx = np.array([np.nanmean((y - m)[kw == q]) for q in range(1, 5)])
dk = seasonal_decompose(y, model="additive", period=4)
print("(2) dekomposisi klasik, NumPy dan statsmodels:")
print("    tren t = 3..10:")
print("    NumPy      :", f(m[2:6]))
print("                ", f(m[6:10]))
print("    statsmodels:", f(dk.trend[2:6]))
print("                ", f(dk.trend[6:10]))
print("    indeks NumPy      :", f(idx))
print("    indeks statsmodels:", f(dk.seasonal[:4]))
