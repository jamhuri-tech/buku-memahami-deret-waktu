"""Memeriksa setiap bilangan Contoh Soal dan hitungan teks Bab 11."""
from fractions import Fraction as Fr

import numpy as np

from bab01_data import deret_musiman
from bab02_acf import acf_sampel
from bab06_arma import acf_arma

# MA(1) x MA(1)_4 dengan theta = Theta = 1/2
th = TH = Fr(1, 2)
w = [Fr(1), th, Fr(0), Fr(0), TH, th * TH]
assert w == [1, Fr(1, 2), 0, 0, Fr(1, 2), Fr(1, 4)]
g = [sum(w[j] * w[j + h] for j in range(6 - h)) for h in range(6)]
assert g[0] == Fr(25, 16) == (1 + th**2) * (1 + TH**2)
assert g[1] == g[4] == Fr(5, 8) and g[3] == g[5] == Fr(1, 4) and g[2] == 0
assert g[1] / g[0] == Fr(2, 5) and g[3] / g[0] == Fr(4, 25)
assert Fr(2, 5) * Fr(2, 5) == Fr(4, 25)
assert np.allclose(acf_arma([], [float(v) for v in w[1:]], 5)[1:],
                   [0.4, 0, 0.16, 0.4, 0.16])

# random walk musiman pada data musiman
y = deret_musiman()
z = y[4:] - y[:-4]
assert list(z) == [4, 4, 1, 7, 4, 4, 7, 1] and z.mean() == 4
assert list(z - 4) == [0, 0, -3, 3, 0, 0, 3, -3]
assert np.mean((z - 4) ** 2) == 4.5 and round(np.sqrt(4.5), 2) == 2.12
assert list(y[8:] + 4) == [14, 19, 24, 17]
t = np.arange(13, 17)
assert list(1 + t + np.array([0, 4, 7, 1])) == [14, 19, 23, 18]
assert np.allclose(acf_sampel(z, 4)[1:], [-0.5, 0, 0.25, -0.5])
# MA musiman Theta = -1: rho(4) = -1/2
assert Fr(-1, 1 + 1) == Fr(-1, 2)
print("Contoh Soal Bab 11: semua bilangan cocok")

# diferensi musiman berlebih
d = np.array([0, 0, -3, 3, 0, 0, 3, -3])
assert d @ d == 36 and d[:-1] @ d[1:] == -18 and d[:-4] @ d[4:] == -18
