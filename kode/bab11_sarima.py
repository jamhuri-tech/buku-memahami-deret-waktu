"""Bab 11: model musiman multiplikatif dan data musiman."""
import warnings

import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX

from bab01_data import deret_musiman
from bab02_acf import acf_sampel

warnings.filterwarnings("ignore")


def baris(v, n=4):
    return " ".join(f"{np.round(x, 10) + 0.0:7.{n}f}" for x in v)


th, TH, s = 0.5, 0.5, 4
ma = np.convolve([1, th], np.r_[1, np.zeros(s - 1), TH])
print("(1) MA(1) x MA(1)_4, theta = Theta = 0.5:")
print("    koefisien MA lag 0..5:", " ".join(f"{v:.2f}" for v in ma))
print("    rho(1..6):", baris(ArmaProcess([1], ma).acf(7)[1:]))

y = deret_musiman()
z = y[4:] - y[:-4]
print("(2) data musiman: diferensi musiman dan ramalan t = 13..16")
print("    nabla_4 y      :", z.astype(int))
print("    r(1..4)        :", baris(acf_sampel(z, 4)[1:]))
fit = SARIMAX(y, order=(0, 0, 0), seasonal_order=(0, 1, 0, 4),
              trend="c").fit(disp=False)
f = fit.get_forecast(4)
t = np.arange(13, 17)
reg = 1 + t + np.array([0, 4, 7, 1])
print("    SARIMAX const  :", f"{fit.params[0]:.3f}")
print("    ramalan SARIMA :", baris(f.predicted_mean, 3))
print("    ramalan regresi:", baris(reg, 3))
print("    SE SARIMA      :", baris(f.se_mean, 3))
