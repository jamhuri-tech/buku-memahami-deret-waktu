"""Memeriksa setiap bilangan Contoh Soal dan hitungan teks Bab 9."""
from fractions import Fraction as Fr

import numpy as np

from bab01_data import deret_mini
from bab06_arma import bobot_psi
from bab07_estimasi import galat_ma1
from bab09_ramalan import galat_baku_ramalan

# AR(2) data mini
psi = [Fr(1), Fr(1, 2)]
for j in range(2, 4):
    psi.append(Fr(1, 2) * psi[j - 1] - Fr(1, 2) * psi[j - 2])
assert psi[1:] == [Fr(1, 2), Fr(-1, 4), Fr(-3, 8)]
assert np.allclose(bobot_psi([0.5, -0.5], [], 3), [float(v) for v in psi])
v = np.cumsum([float(p) ** 2 for p in psi])
assert np.allclose(v[:3], [1, 5 / 4, 21 / 16])
assert round(np.sqrt(5) / 2, 4) == 1.1180 and round(np.sqrt(21) / 4, 4) == 1.1456
assert np.allclose(galat_baku_ramalan([0.5, -0.5], [], 1, 3),
                   [1, np.sqrt(5) / 2, np.sqrt(21) / 4])
assert (round(3.5 - 1.96, 2), round(3.5 + 1.96, 2)) == (1.54, 5.46)
h2 = 1.96 * np.sqrt(5) / 2
assert (round(4.25 - h2, 2), round(4.25 + h2, 2)) == (2.06, 6.44)
assert round(np.sqrt(1.5), 4) == 1.2247

# MA(1) theta = 0.3
e, _ = galat_ma1(deret_mini() - 4, 0.3)
assert np.allclose(np.round(e, 4), [-2, 1.6, 1.52, 0.544, -1.1632, 0.3490,
                                    -0.1047, -0.9686])
assert round(4 + 0.3 * e[-1], 4) == 3.7094
assert round(np.sqrt(1.09), 4) == 1.0440

# random walk lawan AR(2)
assert (round(3 - 1.96 * 2, 2), round(3 + 1.96 * 2, 2)) == (-0.92, 6.92)
assert (round(3 - 1.96 * 4, 2), round(3 + 1.96 * 4, 2)) == (-4.84, 10.84)
se = galat_baku_ramalan([0.5, -0.5], [], 1, 16)
assert round(se[3], 4) == 1.2055 and round(se[15], 4) == 1.2247
assert round(1.96 * se[3], 2) == 2.36 and round(1.96 * se[15], 2) == 2.40
assert 7.84 / 2.40 > 3
assert np.allclose(galat_baku_ramalan([1.0], [], 1, 16)[[3, 15]], [2, 4])
print("Contoh Soal Bab 9: semua bilangan cocok")
