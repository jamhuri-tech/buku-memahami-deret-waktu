"""Memeriksa setiap bilangan Contoh Soal dan hitungan teks Bab 13."""
from fractions import Fraction as Fr

import numpy as np
from scipy import stats

from bab01_data import deret_mini
from bab13_evaluasi import ukuran, uji_dm

y = deret_mini()
yi = [int(v) for v in y]
nyata = yi[4:]
rata = [Fr(sum(yi[:T]), T) for T in range(4, 8)]
e_r = [Fr(a) - b for a, b in zip(nyata, rata)]
e_n = [Fr(yi[T] - yi[T - 1]) for T in range(4, 8)]
assert e_r == [Fr(-3, 2), Fr(-1, 5), Fr(-1, 6), Fr(-8, 7)]
assert e_n == [-2, 1, 0, -1] and nyata == [3, 4, 4, 3]
sa = sum(abs(v) for v in e_r)
assert round(float(sa), 4) == 3.0095 and round(float(sa) / 4, 4) == 0.7524
sq = sum(v * v for v in e_r)
assert round(float(sq), 4) == 3.6239
assert round(np.sqrt(float(sq) / 4), 4) == 0.9518
assert round(np.sqrt(6 / 4), 4) == 1.2247
mr = 100 / 4 * sum(abs(e) / a for e, a in zip(e_r, nyata))
assert round(float(mr), 1) == 24.3
assert 100 / 4 * float(sum(abs(e) / a for e, a in zip(e_n, nyata))) == 31.25
skala = Fr(9, 7)
assert sum(abs(v) for v in np.diff(yi)) == 9 and round(9 / 7, 4) == 1.2857
assert round(0.7524 / (9 / 7), 3) == 0.585 and round(7 / 9, 3) == 0.778
u = ukuran([float(v) for v in e_r], nyata, 9 / 7)
assert round(u[3], 4) == 0.5852

d = [a * a - b * b for a, b in zip(e_r, e_n)]
assert [round(float(v), 4) for v in d] == [-1.75, -0.96, 0.0278, 0.3061]
db = sum(d) / 4
assert round(float(db), 4) == -0.5940
g0 = sum((v - db) ** 2 for v in d) / 4
assert round(float(g0), 4) == 0.6668
dm = float(db) / np.sqrt(float(g0) / 4)
assert round(dm, 3) == -1.455 and round(dm * np.sqrt(0.75), 3) == -1.260
assert round(2 * stats.t.sf(1.26, 3), 2) == 0.30
r = uji_dm([float(v) for v in e_r], [float(v) for v in e_n])
assert np.isclose(r[0], dm)
print("Contoh Soal Bab 13: semua bilangan cocok")
