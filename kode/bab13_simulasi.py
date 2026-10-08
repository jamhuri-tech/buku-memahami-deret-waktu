"""Bab 13: evaluasi asal bergulir pada 200 deret AR(2) simulasi."""
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess

from bab01_data import BENIH, kuadrat_terkecil, ramal
from bab13_evaluasi import uji_dm

rng = np.random.default_rng(BENIH)
proses = ArmaProcess([1, -0.5, 0.5])
R, T0, T = 200, 150, 200
model = ("AR(2)", "AR(10)", "rata-rata", "naif")
rmse_uji = {k: [] for k in model}
rmse_latih = {"AR(2)": [], "AR(10)": []}
tolak = {"rata-rata": 0, "AR(10)": 0}
for _ in range(R):
    y = 4 + proses.generate_sample(T, burnin=200, distrvs=rng.standard_normal)
    galat = {k: [] for k in model}
    for t in range(T0, T):
        for nama, p in (("AR(2)", 2), ("AR(10)", 10)):
            w = kuadrat_terkecil(y[:t], p)
            galat[nama].append(y[t] - ramal(y[:t], w, 1)[0])
        galat["rata-rata"].append(y[t] - y[:t].mean())
        galat["naif"].append(y[t] - y[t - 1])
    for k in model:
        rmse_uji[k].append(np.sqrt(np.mean(np.square(galat[k]))))
    for nama, p in (("AR(2)", 2), ("AR(10)", 10)):
        w = kuadrat_terkecil(y[:T0], p)
        X = np.column_stack([np.ones(T0 - p)] +
                            [y[p - j:T0 - j] for j in range(1, p + 1)])
        rmse_latih[nama].append(np.sqrt(np.mean((y[p:T0] - X @ w) ** 2)))
    for lawan in tolak:
        tolak[lawan] += uji_dm(galat["AR(2)"], galat[lawan])[2] < 0.05
print(f"(3) {R} deret AR(2), latih 150, asal bergulir T = 150..199:")
print("    model       RMSE latih  RMSE uji (rata-rata)")
for k in model:
    lt = f"{np.mean(rmse_latih[k]):8.4f}" if k in rmse_latih else "       -"
    print(f"    {k:10s}  {lt}  {np.mean(rmse_uji[k]):8.4f}")
for lawan, n in tolak.items():
    print(f"    DM AR(2) lawan {lawan:9s} menolak pada 5%: {n / R:.3f}")
