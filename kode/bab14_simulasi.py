"""Bab 14: regresi dengan galat AR(1), GARCH(1,1), fitur lag dan ML."""
import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.regression.linear_model import GLSAR, OLS
from statsmodels.tsa.arima_process import ArmaProcess

from bab01_data import BENIH, kuadrat_terkecil, rancang, tabel_lag
from bab02_acf import acf_sampel
from bab14_perluasan import garch_varians

warnings.filterwarnings("ignore")
rng = np.random.default_rng(BENIH)

# (4) regresi dengan galat AR(1)
R, T, b1 = 500, 100, 1.0
ar = ArmaProcess([1, -0.8])
kena_ols = kena_gls = 0
for _ in range(R):
    x = ar.generate_sample(T, burnin=100, distrvs=rng.standard_normal)
    u = ar.generate_sample(T, burnin=100, distrvs=rng.standard_normal)
    y = 2 + b1 * x + u
    X = np.column_stack([np.ones(T), x])
    lo, hi = OLS(y, X).fit().conf_int()[1]
    kena_ols += lo <= b1 <= hi
    lo, hi = GLSAR(y, X, rho=1).iterative_fit(maxiter=10).conf_int()[1]
    kena_gls += lo <= b1 <= hi
print(f"(4) y = 2 + x + u, x dan u AR(1) 0.8, T = {T}, R = {R}:")
print(f"    cakupan selang 95% beta_1: OLS {kena_ols / R:.3f},"
      f" GLSAR {kena_gls / R:.3f}")

# (5) GARCH(1,1)
T = 3000
om, al, be = 0.1, 0.1, 0.8
z = rng.standard_normal(T)
e, s2 = np.zeros(T), np.zeros(T)
s2[0] = om / (1 - al - be)
e[0] = np.sqrt(s2[0]) * z[0]
for t in range(1, T):
    s2[t] = om + al * e[t - 1] ** 2 + be * s2[t - 1]
    e[t] = np.sqrt(s2[t]) * z[t]


def nll(p):
    o, a, b = p
    if o <= 0 or a < 0 or b < 0 or a + b >= 1:
        return 1e10
    v = garch_varians(e, o, a, b, np.var(e))
    return 0.5 * np.sum(np.log(v) + e**2 / v)


hasil = minimize(nll, [0.05, 0.05, 0.9], method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000})
k = np.mean(e**4) / np.mean(e**2) ** 2
print(f"(5) GARCH(1,1) simulasi, T = {T}:")
print(f"    r(1) e = {acf_sampel(e, 1)[1]:.3f},"
      f" r(1) e^2 = {acf_sampel(e**2, 1)[1]:.3f}, kurtosis = {k:.2f}")
print("    MLE (omega, alpha, beta) =",
      " ".join(f"{v:.3f}" for v in hasil.x))

# (6) fitur lag: AR(2) linear lawan gradient boosting
y = 4 + ArmaProcess([1, -0.5, 0.5]).generate_sample(
    1000, burnin=200, distrvs=rng.standard_normal)
t, F, target = tabel_lag(y, 10)
latih = slice(0, 790)
uji = slice(790, None)
w = np.linalg.lstsq(rancang(F[latih, :2]), target[latih], rcond=None)[0]
e_ar = target[uji] - rancang(F[uji, :2]) @ w
gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                   random_state=BENIH).fit(F[latih],
                                                           target[latih])
e_gb = target[uji] - gb.predict(F[uji])
print("(6) AR(2) simulasi T = 1000, 200 titik uji terakhir:")
print(f"    RMSE AR(2) linear    : {np.sqrt(np.mean(e_ar**2)):.4f}")
print(f"    RMSE boosting 10 lag : {np.sqrt(np.mean(e_gb**2)):.3f}")
