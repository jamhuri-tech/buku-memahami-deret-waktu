"""Memeriksa setiap bilangan Contoh Soal dan hitungan teks Bab 7."""
from fractions import Fraction as Fr

import numpy as np

from bab01_data import deret_mini, rancang, tabel_lag
from bab07_estimasi import galat_ma1, gauss_newton_ma1, loglik_ar1

y = deret_mini()
d = y - 4

# satu langkah Gauss-Newton dari 0
e, g = galat_ma1(d, 0.0)
assert np.allclose(e, d) and e @ e == 12
assert np.allclose(g, np.r_[0, -d[:-1]])
assert g @ e == -1 and g @ g == 11
assert np.isclose(gauss_newton_ma1(d, 0.0, 2)[1][0], 1 / 11)
assert round(1 / 11, 4) == 0.0909
jalur = gauss_newton_ma1(d, 0.0, 30)
assert round(jalur[-1][0], 3) == 0.311 and round(jalur[-1][1], 3) == 11.589

# log-kemungkinan AR(1) eksak di phi = 1/2, mu = 4, s2 = 1
q = Fr(3, 4) * 4 + sum(Fr(int(a)) ** 2 for a in [0])  # suku pertama
galat = [d[t] - 0.5 * d[t - 1] for t in range(1, 8)]
assert np.allclose(galat, [2, 1.5, 0, -1.5, 0.5, 0, -1])
assert np.isclose(np.sum(np.square(galat)), 9.75)
assert np.isclose(3 + 9.75, 12.75)
assert round(-4 * np.log(2 * np.pi), 4) == -7.3515
assert round(0.5 * np.log(0.75), 4) == -0.1438
assert round(loglik_ar1(y, 0.5, 4, 1), 3) == -13.870

# maksimum profil: bersyarat 1/11, eksak sekitar 0.12 (mu = 4)
ph = np.linspace(-0.9, 0.9, 18001)
prof = []
for p in ph:
    qq = (1 - p**2) * d[0] ** 2 + np.sum((d[1:] - p * d[:-1]) ** 2)
    prof.append(-4 * np.log(2 * np.pi * qq / 8) - 4 + 0.5 * np.log(1 - p**2))
assert round(ph[int(np.argmax(prof))], 2) == 0.12
assert np.isclose(np.sum(d[1:] * d[:-1]) / np.sum(d[:-1] ** 2), 1 / 11)

# galat baku AR(2)
t, F, target = tabel_lag(y, 2)
X = rancang(F)
A = X.T @ X
assert round(np.linalg.det(A)) == 356
assert 6 * 115 - 25**2 == 65 and 6 * 127 - 27**2 == 33
V = np.linalg.inv(A)
assert np.isclose(V[1, 1], 65 / 356) and np.isclose(V[2, 2], 33 / 356)
assert round(np.sqrt(65 / 356), 4) == 0.4273
assert round(np.sqrt(33 / 356), 4) == 0.3045
assert round(-0.5 / np.sqrt(33 / 356), 2) == -1.64
assert round(np.sqrt(0.5), 2) == 0.71
G = 1.5 * np.array([[1, 1 / 3], [1 / 3, 1]])
assert np.allclose(np.sqrt(np.diag(np.linalg.inv(G)) / 6), 0.3536, atol=1e-4)

# asimtotik MA(1)
assert round(np.sqrt(0.75 / 50), 4) == 0.1225
print("Contoh Soal Bab 7: semua bilangan cocok")
