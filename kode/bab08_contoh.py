"""Memeriksa setiap bilangan Contoh Soal dan hitungan teks Bab 8."""
from fractions import Fraction as Fr

import numpy as np

from bab01_data import deret_mini

y = deret_mini()
yi = [int(v) for v in y]
dy = [yi[t] - yi[t - 1] for t in range(1, 8)]
assert dy == [3, 1, -1, -2, 1, 0, -1] and sum(dy) == 1

# diferensi berlebih
m = Fr(1, 7)
dev7 = [7 * v - 1 for v in dy]
assert dev7 == [20, 6, -8, -15, 6, -1, -8]
assert sum(v * v for v in dev7) == 826
assert sum(dev7[i] * dev7[i + 1] for i in range(6)) == 104
assert Fr(826, 49 * 7) == Fr(118, 49) and round(118 / 49, 3) == 2.408
assert Fr(17, 7) - m * m == Fr(118, 49)
assert round(104 / 826, 3) == 0.126
assert np.isclose(np.var(np.diff(y)), 118 / 49)
# white noise didiferensi: rho(1) = -1/2
assert Fr(-1, 1 + 1) == Fr(-1, 2)

# Dickey-Fuller dengan konstanta
x = yi[:-1]
Sxx = sum(v * v for v in x) - 7 * Fr(29, 7) ** 2
Sxy = sum(a * b for a, b in zip(x, dy)) - 7 * Fr(29, 7) * Fr(1, 7)
assert sum(x) == 29 and sum(v * v for v in x) == 131
assert sum(a * b for a, b in zip(x, dy)) == -6
assert Sxx == Fr(76, 7) and Sxy == Fr(-71, 7)
d = Sxy / Sxx
assert d == Fr(-71, 76) and round(float(d), 4) == -0.9342
sst = Fr(17) - Fr(1, 7)
assert sst == Fr(118, 7) and Sxy**2 / Sxx == Fr(5041, 532)
jkg = sst - Sxy**2 / Sxx
assert jkg == Fr(3927, 532)
s2 = jkg / 5
assert round(float(s2), 4) == 1.4763
se = np.sqrt(float(s2 / Sxx))
assert round(se, 4) == 0.3687 and round(float(d) / se, 2) == -2.53
assert round(1 + float(d), 3) == 0.066

# KPSS
S = np.cumsum(y - 4)
assert list(S) == [-2, -1, 1, 2, 1, 1, 1, 0] and S @ S == 13
assert Fr(13, 64) / Fr(3, 2) == Fr(13, 96) and round(13 / 96, 4) == 0.1354
print("Contoh Soal Bab 8: semua bilangan cocok")
