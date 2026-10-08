"""Memeriksa setiap bilangan Contoh Soal Bab 2."""
from fractions import Fraction as Fr

import numpy as np
from scipy import stats

from bab01_data import deret_mini
from bab02_acf import acf_sampel, autokov, ljung_box

y = deret_mini()
T = 8
d = [int(v) - 4 for v in y]

# Contoh Soal 2.1: random walk, Cov = min(s, t) sigma^2
s2 = 1.0
for s_, t_ in [(3, 5), (5, 5), (99, 100)]:
    A = np.tril(np.ones((t_, t_)))       # y = A eps
    C = A @ A.T * s2
    assert C[s_ - 1, t_ - 1] == min(s_, t_)
assert round(np.sqrt(99 / 100), 3) == 0.995

# Contoh Soal 2.2: y = e_t + e_{t-1}/2
th = Fr(1, 2)
g0, g1 = 1 + th**2, th
assert (g0, g1) == (Fr(5, 4), Fr(1, 2)) and g1 / g0 == Fr(2, 5)
assert th / (1 + th**2) == Fr(2, 5)

# Contoh Soal 2.3: c(h) dan r(h) data mini
assert d == [-2, 1, 2, 1, -1, 0, 0, -1]
jum = [sum(d[t] * d[t + h] for t in range(T - h)) for h in range(8)]
assert jum[:4] == [12, 1, -5, -2]
c = [Fr(j, T) for j in jum]
assert c[:4] == [Fr(3, 2), Fr(1, 8), Fr(-5, 8), Fr(-1, 4)]
r = [x / c[0] for x in c]
assert r[1:4] == [Fr(1, 12), Fr(-5, 12), Fr(-1, 6)]
assert np.allclose(autokov(y, 3), [float(x) for x in c[:4]])

# Matriks autokovarians 3 x 3 dan bentuk D'D/T
G = np.array([[float(c[abs(i - j)]) for j in range(3)] for i in range(3)])
assert np.isclose(np.linalg.det(G), 697 / 256)
assert np.all(np.linalg.eigvalsh(G) > 0)
D = np.zeros((T + 2, 3))
for j in range(3):
    D[j:j + T, j] = d
assert np.allclose(D.T @ D / T, G)

# Contoh Soal 2.4: jumlah r(1..T-1) = -1/2
assert r[1:] == [Fr(1, 12), Fr(-5, 12), Fr(-1, 6), Fr(1, 12), Fr(-1, 6),
                 Fr(-1, 12), Fr(1, 6)]
assert sum(r[1:]) == Fr(-1, 2)
rng = np.random.default_rng(1)
for _ in range(5):
    z = rng.standard_normal(30)
    assert np.isclose(acf_sampel(z, 29)[1:].sum(), -0.5)

# Contoh Soal 2.5: batas white noise dan Bartlett
assert round(1.96 / np.sqrt(8), 4) == 0.6930 and round(np.sqrt(8), 4) == 2.8284
vb = Fr(1, 8) * (1 + 2 * (r[1]**2 + r[2]**2))
assert vb == Fr(49, 288) and 1 + 2 * Fr(26, 144) == Fr(196, 144)
assert round(float(vb), 4) == 0.1701
assert round(np.sqrt(float(vb)), 4) == 0.4125
assert round(1.96 * np.sqrt(float(vb)), 4) == 0.8085
assert round(1 / 6, 4) == 0.1667 and round(5 / 12, 4) == 0.4167

# Contoh Soal 2.6: Ljung-Box dan Box-Pierce, H = 2
Q = T * (T + 2) * (r[1]**2 / 7 + r[2]**2 / 6)
assert Fr(1, 144) / 7 == Fr(1, 1008) and Fr(25, 144) / 6 == Fr(25, 864)
assert 6048 == 6 * 1008 == 7 * 864
assert Q == Fr(905, 378) and round(float(Q), 4) == 2.3942
assert T * (r[1]**2 + r[2]**2) == Fr(13, 9)
assert round(float(Q) / 2, 4) == 1.1971
p = np.exp(-float(Q) / 2)
assert round(p, 4) == 0.3021 and np.isclose(p, stats.chi2.sf(float(Q), 2))
assert np.isclose(ljung_box(y, 2)[0], float(Q))
print("Contoh Soal Bab 2: semua bilangan cocok")
