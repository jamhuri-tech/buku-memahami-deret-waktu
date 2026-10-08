"""Bab 10: frekuensi pilihan orde AIC, AICc, BIC; derajat bebas
Ljung-Box pada residu."""
import numpy as np
from scipy import stats
from statsmodels.tsa.arima_process import ArmaProcess

from bab01_data import BENIH, rancang, tabel_lag
from bab02_acf import ljung_box
from bab10_diagnostik import kriteria

rng = np.random.default_rng(BENIH)
proses = ArmaProcess([1, -0.5, 0.5])
R, T, PMAKS = 1000, 100, 6
pilih = np.zeros((3, PMAKS + 1), dtype=int)
tolak_h, tolak_hk = 0, 0
for _ in range(R):
    y = proses.generate_sample(T, burnin=200, distrvs=rng.standard_normal)
    t, F, target = tabel_lag(y, PMAKS)
    m = len(target)
    nilai = []
    for p in range(PMAKS + 1):
        X = rancang(F[:, :p]) if p else np.ones((m, 1))
        e = target - X @ np.linalg.lstsq(X, target, rcond=None)[0]
        nilai.append(kriteria(e @ e, m, p + 1))
        if p == 2:
            Q = ljung_box(e, 10)[0]
            tolak_h += stats.chi2.sf(Q, 10) < 0.05
            tolak_hk += stats.chi2.sf(Q, 8) < 0.05
    for j in range(3):
        pilih[j, np.argmin([v[j] for v in nilai])] += 1
print(f"(3) orde terpilih, AR(2) benar, T = {T}, {R} deret:")
print("    p     :", " ".join(f"{p:5d}" for p in range(PMAKS + 1)))
for j, nama in enumerate(("AIC ", "AICc", "BIC ")):
    print(f"    {nama}  :", " ".join(f"{v / R:5.3f}" for v in pilih[j]))
print("(4) Ljung-Box H = 10 pada residu AR(2), taraf 5%:")
print(f"    db = 10: menolak {tolak_h / R:.3f};"
      f" db = 8: menolak {tolak_hk / R:.3f}")
