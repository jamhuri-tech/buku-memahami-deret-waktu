"""Bab 8: regresi palsu, sebaran Dickey-Fuller, ADF dan KPSS."""
import warnings

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

from bab01_data import BENIH

warnings.filterwarnings("ignore")
rng = np.random.default_rng(BENIH)
R, T = 1000, 100

tol, r2 = 0, []
for _ in range(R):
    a = np.cumsum(rng.standard_normal(T))
    b = np.cumsum(rng.standard_normal(T))
    X = np.column_stack([np.ones(T), a])
    w = np.linalg.solve(X.T @ X, X.T @ b)
    e = b - X @ w
    se = np.sqrt(e @ e / (T - 2) * np.linalg.inv(X.T @ X)[1, 1])
    tol += abs(w[1] / se) > 1.96
    r2.append(1 - e @ e / np.sum((b - b.mean()) ** 2))
print(f"(3) regresi palsu, dua random walk bebas, T = {T}, R = {R}:")
print(f"    |t| > 1.96: {tol / R:.3f};  median R^2 = {np.median(r2):.3f}")

tau = []
for _ in range(R):
    y = np.cumsum(rng.standard_normal(T))
    tau.append(adfuller(y, maxlag=0, regression="c", autolag=None)[0])
tau = np.array(tau)
print(f"(4) tau Dickey-Fuller di bawah H0 (random walk), T = {T}:")
print(f"    kuantil 5% = {np.quantile(tau, 0.05):.2f},"
      f" rata-rata = {tau.mean():.2f}")
print(f"    proporsi tau < -1.645: {np.mean(tau < -1.645):.3f},"
      f" tau < -2.89: {np.mean(tau < -2.89):.3f}")

t = np.arange(1, T + 1)
rw = np.cumsum(0.2 + rng.standard_normal(T))
u = np.zeros(T)
eps = rng.standard_normal(T)
for i in range(1, T):
    u[i] = 0.5 * u[i - 1] + eps[i]
deret = {"random walk + drift": rw, "tren + AR(1)": 0.2 * t + u}
print("(5) ADF (konstanta + tren) dan KPSS (tren), nilai-p:")
for nama, y in deret.items():
    p_adf = adfuller(y, regression="ct", autolag="AIC")[1]
    p_kpss = kpss(y, regression="ct", nlags="auto")[1]
    print(f"    {nama:20s} ADF {p_adf:.3f}  KPSS {p_kpss:.3f}")
