"""Memeriksa setiap bilangan Contoh Soal dan hitungan teks Bab 3."""
from fractions import Fraction as Fr

import numpy as np

from bab01_data import deret_musiman

y = deret_musiman()
yi = [int(v) for v in y]
t = np.arange(1, 13)
kw = (t - 1) % 4 + 1
assert [sum(yi[4 * k:4 * k + 4]) for k in range(3)] == [26, 42, 58]

# tren linear
assert sum(yi) == 126 and int(t @ y) == 971
assert ((t - 6.5) ** 2).sum() == 143 == 12 * (144 - 1) / 12
assert 971 - 12 * 6.5 * 10.5 == 152
w1 = Fr(152, 143)
w0 = Fr(21, 2) - Fr(13, 2) * w1
assert round(float(w1), 4) == 1.0629 and round(float(w0), 4) == 3.5909
X = np.column_stack([np.ones(12), t])
w = np.linalg.solve(X.T @ X, X.T @ y)
assert np.allclose(w, [float(w0), float(w1)])
e = y - X @ w
assert [round(e[kw == q].mean(), 2) for q in range(1, 5)] == [-2.91, 1.03,
                                                              3.97, -2.09]

# tren dan dummy
D = np.column_stack([(kw == q).astype(float) for q in (2, 3, 4)])
X = np.column_stack([np.ones(12), t, D])
wd = np.array([1, 1, 4, 7, 1.0])
yh = X @ wd
assert list(yh) == [2, 7, 11, 6, 6, 11, 15, 10, 10, 15, 19, 14]
e = y - yh
assert list(e) == [0, 0, 1, -1, 0, 0, -2, 2, 0, 0, 1, -1]
assert list(t * e) == [0, 0, 3, -4, 0, 0, -14, 16, 0, 0, 11, -12]
assert np.allclose(X.T @ e, 0) and np.linalg.matrix_rank(X) == 5
assert e @ e == 12 and round(12 / 7, 4) == 1.7143
assert np.allclose(np.linalg.lstsq(X, y, rcond=None)[0], wd)
assert np.mean([0, 4, 7, 1]) == 3

# rata-rata bergerak 2x4
b = [Fr(1, 8), Fr(1, 4), Fr(1, 4), Fr(1, 4), Fr(1, 8)]
assert sum(b) == 1 and sum(bk * k for bk, k in zip(b, range(-2, 3))) == 0
S = [Fr(-3), Fr(1), Fr(4), Fr(-2)]
for tt in range(3, 11):
    lin = [Fr(5) + 2 * (tt + k) + S[(tt + k - 1) % 4] for k in range(-2, 3)]
    assert sum(bk * v for bk, v in zip(b, lin)) == 5 + 2 * tt
m = {tt: sum(bk * yi[tt - 1 + k] for bk, k in zip(b, range(-2, 3)))
     for tt in range(3, 11)}
assert (2 + 14 + 24 + 10 + 6, 7 + 24 + 10 + 12 + 11) == (56, 64)
assert m[3] == 7 and m[4] == 8
assert [8 * m[k] for k in range(3, 11)] == [56, 64, 69, 77, 88, 96, 107, 115]

# indeks musiman
d = {tt: yi[tt - 1] - m[tt] for tt in m}
assert (d[5], d[9], d[6], d[10]) == (Fr(-21, 8), Fr(-27, 8), Fr(11, 8),
                                     Fr(5, 8))
assert (d[3], d[7], d[4], d[8]) == (5, 2, -3, 0)
idx = [(d[q + 4] + d[q + 8]) / 2 if q <= 2 else (d[q] + d[q + 4]) / 2
       for q in range(1, 5)]
assert idx == [-3, 1, Fr(7, 2), Fr(-3, 2)] and sum(idx) == 0
assert [yi[k] - idx[k] for k in range(4)] == [5, 6, Fr(17, 2), Fr(13, 2)]

# diferensi musiman
d4 = [yi[k] - yi[k - 4] for k in range(4, 12)]
assert d4 == [4, 4, 1, 7, 4, 4, 7, 1] and sum(d4) == 32
assert (58 - 26) / 8 == 4

# musiman multiplikatif
assert (140 - 70, 280 - 140) == (70, 140)
assert np.isclose(np.log(1.4 * 100) - np.log(0.7 * 100), np.log(2))
assert np.isclose(np.log(1.4 * 200) - np.log(0.7 * 200), np.log(2))
assert round(np.log(2), 4) == 0.6931
print("Contoh Soal Bab 3: semua bilangan cocok")
