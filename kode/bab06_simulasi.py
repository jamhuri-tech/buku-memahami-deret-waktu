"""Bab 6: memulihkan inovasi dari data, MA(1) dapat dibalik atau tidak."""
import numpy as np

from bab01_data import BENIH

rng = np.random.default_rng(BENIH)
T = 60
eps = rng.standard_normal(T + 1)
print("(3) |e_t - eps_t| dengan e_t = y_t - theta e_{t-1}, e_0 = 0:")
print("    t           :        1       10       20       30")
for th in (0.5, 2.0):
    y = eps[1:] + th * eps[:-1]
    e = np.zeros(T)
    sebelum = 0.0
    for t in range(T):
        e[t] = y[t] - th * sebelum
        sebelum = e[t]
    g = np.abs(e - eps[1:])
    print(f"    theta = {th} :", " ".join(f"{g[k]:8.1e}"
                                         for k in (0, 9, 19, 29)))
