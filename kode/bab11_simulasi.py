"""Bab 11: deret bulanan dari model airline dan penaksirannya."""
import warnings

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX

from bab01_data import BENIH

warnings.filterwarnings("ignore")


def deret_airline(T, theta, Theta, s=12, benih=BENIH):
    """y dengan (1 - B)(1 - B^s) y_t = (1 + theta B)(1 + Theta B^s) e_t."""
    rng = np.random.default_rng(benih)
    ma = np.convolve([1, theta], np.r_[1, np.zeros(s - 1), Theta])
    w = ArmaProcess([1], ma).generate_sample(T, distrvs=rng.standard_normal)
    pola = 5 * np.sin(2 * np.pi * np.arange(s + 1) / s)
    y = np.zeros(T)
    y[:s + 1] = 100 + pola
    for t in range(s + 1, T):
        y[t] = y[t - 1] + y[t - s] - y[t - s - 1] + w[t]
    return y


if __name__ == "__main__":
    y = deret_airline(144, -0.4, -0.6)
    fit = SARIMAX(y, order=(0, 1, 1),
                  seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    lb = acorr_ljungbox(fit.resid[13:], lags=[24], model_df=2)
    print("(3) model airline, T = 144, theta = -0.4, Theta = -0.6:")
    print(f"    theta = {fit.params[0]:.3f} (SE {fit.bse[0]:.3f}),"
          f" Theta = {fit.params[1]:.3f} (SE {fit.bse[1]:.3f})")
    print(f"    sigma2 = {fit.params[2]:.3f},"
          f" Ljung-Box H = 24: p = {lb['lb_pvalue'].iloc[0]:.3f}")
