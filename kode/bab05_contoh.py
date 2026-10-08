"""Memeriksa setiap bilangan Contoh Soal dan hitungan teks Bab 5."""
from fractions import Fraction as Fr

import numpy as np

from bab01_data import deret_mini
from bab02_acf import acf_sampel, autokov
from bab04_ar import yule_walker
from bab05_pacf import durbin_levinson, pacf_sampel


def dl_eksak(r, H):
    """Durbin-Levinson dengan pecahan eksak; r = rho(0..H)."""
    phi, alpha, v = [], [], [Fr(1)]
    for h in range(1, H + 1):
        num = r[h] - sum(phi[j] * r[h - 1 - j] for j in range(h - 1))
        den = 1 - sum(phi[j] * r[j + 1] for j in range(h - 1))
        k = num / den
        phi = [phi[j] - k * phi[h - 2 - j] for j in range(h - 1)] + [k]
        alpha.append(k)
        v.append(v[-1] * (1 - k * k))
    return alpha, phi, v


def alpha2(r1, r2):
    return (r2 - r1**2) / (1 - r1**2)


# AR(1) phi = 0.7 dan MA(1) theta = 1/2
assert np.isclose(alpha2(0.7, 0.49), 0)
assert alpha2(Fr(2, 5), Fr(0)) == Fr(-4, 21)
assert round(-4 / 21, 4) == -0.1905
th = Fr(1, 2)
assert -th**2 / (1 + th**2 + th**4) == Fr(-4, 21)          # latihan 2
assert np.allclose(durbin_levinson([1, 0.7, 0.49, 0.343], 3)[0],
                   [0.7, 0, 0])

# AR(2) phi = (1/2, -1/2): rho = 1/3, -1/3, -1/3
r = [Fr(1), Fr(1, 3), Fr(-1, 3), Fr(-1, 3)]
a, phi, v = dl_eksak(r, 3)
assert a == [Fr(1, 3), Fr(-1, 2), Fr(0)]
g0 = Fr(3, 2)
assert g0 * v[1] == Fr(4, 3) and g0 * v[2] == 1
p2 = dl_eksak(r, 2)[1]
assert p2 == [Fr(1, 2), Fr(-1, 2)]

# PACF sampel data mini
y = deret_mini()
rs = [Fr(1), Fr(1, 12), Fr(-5, 12), Fr(-1, 6)]
assert np.allclose(acf_sampel(y, 3), [float(x) for x in rs])
a, phi, v = dl_eksak(rs, 3)
assert a == [Fr(1, 12), Fr(-61, 143), Fr(-70, 697)]
assert alpha2(rs[1], rs[2]) == Fr(-61, 143)
assert dl_eksak(rs, 2)[1][0] == Fr(17, 143)
assert 1716 == 12 * 143
num = Fr(-1, 6) + Fr(85, 1716) + Fr(61, 1716)
den = 1 - Fr(17, 1716) - Fr(305, 1716)
assert num == Fr(-140, 1716) and den == Fr(1394, 1716)
assert num / den == Fr(-70, 697) and round(-70 / 697, 4) == -0.1004
c0 = Fr(3, 2)
assert c0 * v[1] == Fr(143, 96) and c0 * v[2] == Fr(697, 572)
assert np.isclose(yule_walker(y, 2)[1], float(c0 * v[2]))
assert np.isclose(yule_walker(y, 2)[0][1], -61 / 143)
assert np.allclose(pacf_sampel(y, 3), [float(x) for x in a])
assert np.isclose(autokov(y, 0)[0], 1.5)
assert round(1.96 / np.sqrt(8), 2) == 0.69

# batas identifikasi
assert round(1.96 / np.sqrt(200), 4) == 0.1386
assert round(1 / np.sqrt(200), 4) == 0.0707
print("Contoh Soal Bab 5: semua bilangan cocok")
