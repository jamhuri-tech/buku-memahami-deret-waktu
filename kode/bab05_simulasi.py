"""Bab 5: sebaran PACF sampel di atas orde AR, dan contoh identifikasi."""
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess

from bab01_data import BENIH
from bab02_acf import acf_sampel
from bab05_pacf import pacf_sampel

rng = np.random.default_rng(BENIH)
R, T = 2000, 200
ar2 = ArmaProcess([1, -0.5, 0.5])
a = np.array([pacf_sampel(ar2.generate_sample(
    T, burnin=200, distrvs=rng.standard_normal), 5) for _ in range(R)])
print(f"(3) PACF sampel AR(2) phi = (0.5, -0.5), {R} deret, T = {T}:")
print("    h          :      1       2       3       4       5")
print("    rata-rata  :", " ".join(f"{v:7.4f}" for v in a.mean(axis=0)))
print("    simp. baku :", " ".join(f"{v:7.4f}" for v in a.std(axis=0)))
print(f"    1/sqrt(T) = {1 / np.sqrt(T):.4f}")

print(f"(4) dua deret simulasi, T = {T}, batas = {1.96 / np.sqrt(T):.4f}:")
for nama, proses in [("deret A", ArmaProcess([1, -0.6, 0.3])),
                     ("deret B", ArmaProcess([1], [1, 0.7]))]:
    y = proses.generate_sample(T, burnin=200, distrvs=rng.standard_normal)
    print(f"    {nama}  r    :", " ".join(f"{v:6.3f}"
                                        for v in acf_sampel(y, 5)[1:]))
    print("             PACF :", " ".join(f"{v:6.3f}"
                                         for v in pacf_sampel(y, 5)))
