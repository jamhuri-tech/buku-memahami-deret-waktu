"""Memeriksa setiap bilangan Contoh Soal Bab 4."""
from fractions import Fraction as Fr

import numpy as np

from bab01_data import deret_mini
from bab02_acf import autokov
from bab04_ar import acf_ar, yule_walker

# Contoh Soal 4.1: AR(1), c = 1, phi = 0.8, sigma2 = 0.36
c, phi, s2 = 1.0, 0.8, 0.36
assert np.isclose(c / (1 - phi), 5) and np.isclose(s2 / (1 - phi**2), 1)
assert np.isclose(phi**3, 0.512)
assert round(np.log(0.5), 4) == -0.6931 and round(np.log(0.8), 4) == -0.2231
assert round(np.log(0.5) / np.log(0.8), 2) == 3.11
e1 = c + phi * 7
e2 = c + phi * e1
assert np.isclose(e1, 6.6) and np.isclose(e2, 6.28)
assert np.isclose(5 + 2 * phi**2, e2)
assert np.allclose(acf_ar([0.8], 3), 0.8 ** np.arange(4))
# waktu paruh di teks
assert round(np.log(0.5) / np.log(0.9), 1) == 6.6
assert round(np.log(0.5) / np.log(0.3), 2) == 0.58

# Contoh Soal 4.2: akar 1 - z/2 + z^2/2
p1, p2 = Fr(1, 2), Fr(-1, 2)
assert p1 + p2 < 1 and p2 - p1 == -1 and abs(p2) < 1
akar = np.roots([1, -1, 2])
assert np.allclose(sorted(akar, key=np.imag),
                   [(1 - 1j * np.sqrt(7)) / 2, (1 + 1j * np.sqrt(7)) / 2])
assert np.allclose(np.abs(akar), np.sqrt(2)) and round(np.sqrt(2), 4) == 1.4142
assert p1**2 + 4 * p2 < 0

# Contoh Soal 4.3: ACF teoretis
r1 = p1 / (1 - p2)
r2 = p1 * r1 + p2
r = [Fr(1), r1, r2]
for h in range(3, 6):
    r.append(p1 * r[h - 1] + p2 * r[h - 2])
assert r[1:] == [Fr(1, 3), Fr(-1, 3), Fr(-1, 3), Fr(0), Fr(1, 6)]
assert 1 - p1 * r1 - p2 * r2 == Fr(2, 3)
assert np.allclose(acf_ar([0.5, -0.5], 5), [float(v) for v in r])
th = np.arccos(0.5 / (2 * np.sqrt(0.5)))
assert round(np.sqrt(0.5), 4) == 0.7071 and round(np.cos(th), 4) == 0.3536
assert round(th, 4) == 1.2094 and round(2 * np.pi / th, 3) == 5.195
# bentuk A r^h cos(theta h + psi) cocok dengan rekursi
lam = np.roots([1, -0.5, 0.5])
assert np.allclose(np.abs(lam), np.sqrt(0.5))

# Contoh Soal 4.4: Yule-Walker data mini
y = deret_mini()
c = autokov(y, 2)
assert np.allclose(c, [1.5, 0.125, -0.625])
ph1 = Fr(12 * 1 - 1 * (-5), 143)
ph2 = Fr(12 * (-5) - 1 * 1, 143)
assert (ph1, ph2) == (Fr(17, 143), Fr(-61, 143))
assert np.allclose(yule_walker(y, 2)[0], [float(ph1), float(ph2)])
s2 = Fr(3, 2) - ph1 * Fr(1, 8) - ph2 * Fr(-5, 8)
assert s2 == Fr(1394, 1144) == Fr(697, 572) and 1716 - 17 - 305 == 1394
assert round(float(s2), 4) == 1.2185
assert np.isclose(yule_walker(y, 2)[1], float(s2))
ch = 4 * (1 - ph1 - ph2)
assert ch == 4 * Fr(187, 143) and round(float(ch), 4) == 5.2308
print("Contoh Soal Bab 4: semua bilangan cocok")
