"""Bab 4: kuadrat terkecil lawan Yule-Walker pada data simulasi AR(2)."""
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess

from bab01_data import BENIH, kuadrat_terkecil
from bab04_ar import yule_walker

rng = np.random.default_rng(BENIH)
proses = ArmaProcess([1, -0.5, 0.5])
R = 1000
print(f"(3) AR(2) phi = (0.5, -0.5), mu = 4, {R} ulangan:")
print("       T   metode   rata-rata phi1  phi2    RMSE")
for T in (8, 50, 200, 1000):
    hasil = {"KT": [], "YW": []}
    for _ in range(R):
        y = 4 + proses.generate_sample(T, burnin=200,
                                       distrvs=rng.standard_normal)
        hasil["KT"].append(kuadrat_terkecil(y, 2)[1:])
        hasil["YW"].append(yule_walker(y, 2)[0])
    for nama, v in hasil.items():
        v = np.array(v)
        rmse = np.sqrt(np.mean(np.sum((v - [0.5, -0.5]) ** 2, axis=1)))
        print(f"    {T:4d}   {nama}       {v[:, 0].mean():7.4f}"
              f" {v[:, 1].mean():7.4f}  {rmse:6.4f}")
