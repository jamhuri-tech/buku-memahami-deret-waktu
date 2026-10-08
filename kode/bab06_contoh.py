"""Memeriksa setiap bilangan Contoh Soal dan hitungan teks Bab 6."""
from fractions import Fraction as Fr

import numpy as np

from bab06_arma import acf_arma, bobot_pi, bobot_psi

# MA(1): theta dan 1/theta
for th in (0.5, 2.0, 3.0):
    assert np.isclose(th / (1 + th**2), (1 / th) / (1 + 1 / th**2))
assert np.isclose(2 * 1, 0.5 * 4) and np.isclose((1 + 4) * 1, (1 + 0.25) * 4)
th = np.linspace(-5, 5, 1001)
assert np.all(np.abs(th / (1 + th**2)) <= 0.5 + 1e-12)

# momen MA(1) pada data mini
akar = np.sort(np.roots([1, -12, 1]))
assert np.allclose(akar, [6 - np.sqrt(35), 6 + np.sqrt(35)])
assert round(akar[0], 4) == 0.0839 and round(akar[1], 4) == 11.9161
assert np.isclose(akar[0] * akar[1], 1)
assert np.isclose(akar[0] / (1 + akar[0]**2), 1 / 12)
assert round(1.5 / (1 + akar[0]**2), 4) == 1.4895

# MA(2) theta = (1, 1/2)
t = [Fr(1), Fr(1), Fr(1, 2)]
g = [sum(t[j] * t[j + h] for j in range(3 - h)) for h in range(3)]
assert g == [Fr(9, 4), Fr(3, 2), Fr(1, 2)]
assert g[1] / g[0] == Fr(2, 3) and g[2] / g[0] == Fr(2, 9)
z = np.roots([0.5, 1, 1])
assert np.allclose(sorted(z, key=np.imag), [-1 - 1j, -1 + 1j])
assert np.allclose(np.abs(z), np.sqrt(2))
assert np.allclose(acf_arma([], [1, 0.5], 2)[1:], [2 / 3, 2 / 9])

# ARMA(1,1) phi = theta = 1/2
psi = bobot_psi([0.5], [0.5], 4)
assert np.allclose(psi, [1, 1, 0.5, 0.25, 0.125])
f, q = Fr(1, 2), Fr(1, 2)
g0 = (1 + 2 * f * q + q**2) / (1 - f**2)
r1 = (f + q) * (1 + f * q) / (1 + 2 * f * q + q**2)
assert g0 == Fr(7, 3) and r1 == Fr(5, 7) and f * r1 == Fr(5, 14)
assert np.allclose(acf_arma([0.5], [0.5], 2)[1:], [5 / 7, 5 / 14])
assert round(25 / 49, 2) == 0.51

# akar saling menghapus
assert np.allclose(bobot_psi([0.5], [-0.5], 6)[1:], 0)

# bobot pi MA(1)
assert np.allclose(bobot_pi([], [0.5], 4), (-0.5) ** np.arange(5))
print("Contoh Soal Bab 6: semua bilangan cocok")
