"""Memeriksa setiap bilangan Contoh Soal dan hitungan teks Bab 12."""
from fractions import Fraction as Fr

import numpy as np

from bab01_data import deret_mini, deret_musiman
from bab12_smoothing import holt_winters, ses

# SES data mini, alpha = 1/2, l0 = 2 (pecahan eksak)
l, ls, es = Fr(2), [], []
for v in [2, 5, 6, 5, 3, 4, 4, 3]:
    e = v - l
    l = l + e / 2
    es.append(e)
    ls.append(l)
assert es[:7] == [0, 3, Fr(5, 2), Fr(1, 4), Fr(-15, 8), Fr(1, 16),
                  Fr(1, 32)]
assert es[7] == Fr(-63, 64) and round(float(es[7]), 3) == -0.984
assert ls[:5] == [2, Fr(7, 2), Fr(19, 4), Fr(39, 8), Fr(63, 16)]
assert [round(float(v), 3) for v in ls[5:]] == [3.969, 3.984, 3.492]
jkg = sum(e * e for e in es)
assert round(float(jkg), 2) == 19.80
L, E = ses(deret_mini(), 0.5, 2.0)
assert np.allclose(L, [float(v) for v in ls])
# bobot SES menjumlah 1
a, t = 0.3, 10
assert np.isclose(sum(a * (1 - a) ** j for j in range(t)) + (1 - a) ** t, 1)

# Holt-Winters data musiman
z = deret_musiman()
Lh, Bh, Sh, Eh = holt_winters(z, 0.5, 0.5, 0.5, 4.0, 1.0, [-3, 1, 4, -2])
assert np.allclose(Eh[:3], [0, 0, 1]) and np.allclose(Lh[:3], [5, 6, 7.5])
assert np.isclose(Bh[2], 1.25) and np.isclose(Sh[4 + 2], 4.5)
assert np.isclose(Lh[2] + Bh[2] + Sh[3], 6.75) and np.isclose(Eh[3], -1.75)
f = Lh[-1] + Bh[-1] * np.arange(1, 5) + Sh[-4:]
assert round(f[3], 3) == 14.623
print("Contoh Soal Bab 12: semua bilangan cocok")

# kesetaraan SES dan ARIMA(0,1,1)
E = ses(deret_mini(), 0.5, 2.0)[1]
d = np.diff(deret_mini())
assert np.allclose(E[2:5] - 0.5 * E[1:4], d[1:4]) and list(d[1:4]) == [1, -1, -2]
assert [1 + (h - 1) * 0.25 for h in (1, 2, 3)] == [1, 1.25, 1.5]
