"""Bab 7: CSS lawan MLE untuk MA(1), theta = 0.5, T = 50."""
import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess

from bab01_data import BENIH
from bab07_estimasi import gauss_newton_ma1

warnings.filterwarnings("ignore")
rng = np.random.default_rng(BENIH)
R, T, th = 500, 50, 0.5
proses = ArmaProcess([1], [1, th])
css, mle, cak = [], [], []
for _ in range(R):
    y = proses.generate_sample(T, distrvs=rng.standard_normal)
    css.append(gauss_newton_ma1(y - y.mean(), 0.0, 30)[-1][0])
    f = ARIMA(y, order=(0, 0, 1)).fit()
    mle.append(f.params[1])
    lo, hi = f.conf_int()[1]
    cak.append(lo <= th <= hi)
css, mle = np.array(css), np.array(mle)
print(f"(4) MA(1) theta = {th}, T = {T}, {R} deret:")
print(f"    CSS: rata-rata {css.mean():.3f}, simp. baku {css.std():.3f}")
print(f"    MLE: rata-rata {mle.mean():.3f}, simp. baku {mle.std():.3f}")
print(f"    asimtotik sqrt((1 - theta^2)/T) = {np.sqrt((1 - th**2) / T):.4f}")
print(f"    selang 95% MLE memuat theta: {np.mean(cak):.3f}")
