"""Bab 9: cakupan selang ramalan 95% bila parameter ditaksir."""
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess

from bab01_data import BENIH, kuadrat_terkecil, ramal
from bab09_ramalan import galat_baku_ramalan

rng = np.random.default_rng(BENIH)
proses = ArmaProcess([1, -0.5, 0.5])
H = (1, 2, 5, 10)
R = 2000
for T in (30, 200):
    kena_benar = np.zeros(len(H))
    kena_taksir = np.zeros(len(H))
    for _ in range(R):
        z = 4 + proses.generate_sample(T + 10, burnin=200,
                                       distrvs=rng.standard_normal)
        y, masa_depan = z[:T], z[T:]
        f0 = ramal(y, np.array([4, 0.5, -0.5]), 10)
        se0 = galat_baku_ramalan([0.5, -0.5], [], 1.0, 10)
        w = kuadrat_terkecil(y, 2)
        X = np.column_stack([np.ones(T - 2), y[1:-1], y[:-2]])
        e = y[2:] - X @ w
        s2 = e @ e / (T - 2 - 3)
        f1 = ramal(y, w, 10)
        se1 = galat_baku_ramalan(w[1:], [], s2, 10)
        for k, h in enumerate(H):
            kena_benar[k] += abs(masa_depan[h - 1] - f0[h - 1]) \
                <= 1.96 * se0[h - 1]
            kena_taksir[k] += abs(masa_depan[h - 1] - f1[h - 1]) \
                <= 1.96 * se1[h - 1]
    if T == 30:
        print(f"(4) cakupan selang 95% AR(2), {R} deret:")
        print("    h                    :     1     2     5    10")
    print(f"    T = {T:3d}, parameter benar :",
          " ".join(f"{v / R:5.3f}" for v in kena_benar))
    print(f"    T = {T:3d}, parameter taksir:",
          " ".join(f"{v / R:5.3f}" for v in kena_taksir))
