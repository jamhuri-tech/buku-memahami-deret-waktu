"""Bab 8: regresi Dickey-Fuller dan KPSS pada data mini."""
import warnings

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

from bab01_data import deret_mini

warnings.filterwarnings("ignore")
y = deret_mini()

x, dy = y[:-1], np.diff(y)
X = np.column_stack([np.ones(len(x)), x])
b = np.linalg.solve(X.T @ X, X.T @ dy)
e = dy - X @ b
s2 = e @ e / (len(dy) - 2)
se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
adf = adfuller(y, maxlag=0, regression="c", autolag=None)
print("(1) Dickey-Fuller dengan konstanta, data mini:")
print(f"    NumPy      : delta = {b[1]:.4f}, SE = {se:.4f},"
      f" tau = {b[1] / se:.4f}")
print(f"    statsmodels: tau = {adf[0]:.4f}, p = {adf[1]:.3f},"
      f" kritis 5% = {adf[4]['5%']:.3f}")

S = np.cumsum(y - y.mean())
eta = (S @ S) / (len(y) ** 2 * np.mean((y - y.mean()) ** 2))
k = kpss(y, regression="c", nlags=0)
print("(2) KPSS tingkat, lag 0, data mini:")
print(f"    NumPy      : eta = {eta:.4f}")
print(f"    statsmodels: eta = {k[0]:.4f}, kritis 5% = {k[3]['5%']:.3f}")
