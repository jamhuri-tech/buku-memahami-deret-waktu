"""Bab 2: memeriksa batas white noise dan uji Ljung-Box dengan simulasi."""
import numpy as np
from statsmodels.tsa.stattools import acf

from bab01_data import BENIH
from bab02_acf import acf_sampel, ljung_box

rng = np.random.default_rng(BENIH)
R, T, H = 2000, 100, 10

r1 = np.empty(R)
luar = np.empty(R)
tolak = np.empty(R, dtype=bool)
for k in range(R):
    e = rng.standard_normal(T)
    r = acf_sampel(e, 20)
    r1[k] = r[1]
    luar[k] = np.mean(np.abs(r[1:]) > 1.96 / np.sqrt(T))
    tolak[k] = ljung_box(e, H)[1] < 0.05

print(f"(6) white noise, {R} deret, T = {T}:")
print(f"    rata-rata r(1) = {r1.mean():.4f}  (-1/T = {-1 / T:.4f})")
print(f"    simpangan baku r(1) = {r1.std():.4f}"
      f"  (1/sqrt(T) = {1 / np.sqrt(T):.4f})")
print(f"    proporsi |r(h)| > 1.96/sqrt(T), h = 1..20: {luar.mean():.4f}")
print(f"    Ljung-Box H = {H} menolak pada 5%: {tolak.mean():.4f}")

w = np.cumsum(rng.standard_normal(T))
t = np.arange(1, T + 1)
tr = 0.1 * t + rng.standard_normal(T)
print("(7) ACF sampel lag 1, 5, 10:")
for nama, s in [("white noise", rng.standard_normal(T)),
                ("random walk", w), ("tren + noise", tr)]:
    r = acf(s, nlags=10)
    print(f"    {nama:13s} {r[1]:7.4f} {r[5]:7.4f} {r[10]:7.4f}")
