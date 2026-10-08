"""Memeriksa setiap bilangan Contoh Soal dan hitungan teks Bab 10."""
from fractions import Fraction as Fr

import numpy as np
from scipy import stats

from bab01_data import deret_mini, rancang, tabel_lag
from bab10_diagnostik import kriteria

y = deret_mini()
t, F, target = tabel_lag(y, 2)
jkg = []
for p in range(3):
    X = rancang(F[:, :p]) if p else np.ones((6, 1))
    e = target - X @ np.linalg.lstsq(X, target, rcond=None)[0]
    jkg.append(e @ e)
assert np.allclose(jkg, [41 / 6, 188 / 33, 3])
assert Fr(111) - Fr(625, 6) == Fr(41, 6)
assert Fr(41, 6) - Fr(25, 4) / Fr(11, 2) == Fr(188, 33)
dasar = [6 * np.log(j / 6) for j in jkg]
assert [round(v, 3) for v in dasar] == [0.780, -0.311, -4.159]
tab = [kriteria(j, 6, k + 1) for k, j in enumerate(jkg)]
assert [round(v[0], 3) for v in tab] == [2.780, 3.689, 1.841]
assert [round(v[2], 3) for v in tab] == [2.572, 3.273, 1.216]
assert [round(v[1], 3) for v in tab] == [3.780, 7.689, 13.841]
assert [2 * k * (k + 1) / (5 - k) for k in (1, 2, 3)] == [1, 4, 12]
assert round(np.log(6), 3) == 1.792
assert np.argmin([v[0] for v in tab]) == 2 and np.argmin([v[2] for v in tab]) == 2
assert np.argmin([v[1] for v in tab]) == 0

# residu AR(2)
r = [Fr(1, 2), Fr(1, 2), Fr(-1, 2), Fr(1), Fr(-1, 2), Fr(-1)]
assert sum(r) == 0 and sum(v * v for v in r) == 3
s = [sum(r[i] * r[i + h] for i in range(6 - h)) for h in (1, 2, 3)]
assert s == [Fr(-1, 2), Fr(-1, 2), Fr(3, 4)]
rh = [v / 3 for v in s]
assert rh == [Fr(-1, 6), Fr(-1, 6), Fr(1, 4)]
Q = 6 * 8 * sum(rh[h - 1] ** 2 / (6 - h) for h in (1, 2, 3))
assert Q == Fr(8, 5) and 4 + 5 + 15 == 24
assert round(stats.chi2.sf(1.6, 1), 3) == 0.206
m2 = Fr(1, 2)
m4 = sum(v**4 for v in r) / 6
assert m4 == Fr(3, 8) and sum(v**3 for v in r) == 0
K = m4 / m2**2
assert K == Fr(3, 2)
JB = Fr(6, 6) * (K - 3) ** 2 / 4
assert JB == Fr(9, 16) and round(np.exp(-9 / 32), 3) == 0.755
print("Contoh Soal Bab 10: semua bilangan cocok")
