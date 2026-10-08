"""Memeriksa setiap bilangan Contoh Soal dan hitungan teks Bab 1."""
from fractions import Fraction as Fr

import numpy as np

from bab01_data import deret_mini, rancang, ramal, tabel_lag

y = deret_mini()
assert len(y) == 8 and y.sum() == 32 and y.mean() == 4

# diferensi dan lag
d1 = np.diff(y)
d2 = np.diff(y, 2)
assert list(d1) == [3, 1, -1, -2, 1, 0, -1]
assert list(d2) == [-2, -2, -1, 3, -1, -1]
assert y[8 - 3 - 1] == 3                         # B^3 y_8 = y_5
assert y[7] - 2 * y[6] + y[5] == -1 == d2[-1]
assert d1.sum() == y[7] - y[0] == 1

# X'X dan X'y
t, F, target = tabel_lag(y, 2)
X = rancang(F)
assert list(t) == [3, 4, 5, 6, 7, 8]
assert F.tolist() == [[5, 2], [6, 5], [5, 6], [3, 5], [4, 3], [4, 4]]
assert list(target) == [6, 5, 3, 4, 4, 3]
XtX, Xty = X.T @ X, X.T @ target
assert XtX.tolist() == [[6, 27, 25], [27, 127, 113], [25, 113, 115]]
assert Xty.tolist() == [25, 115, 99]
assert round(np.linalg.det(XtX)) == 356

# satu baris t = 3
x3, y3 = X[0], target[0]
e = y3 - x3 @ np.zeros(3)
assert e == 6 and 0.5 * e**2 == 18
assert list(-e * x3) == [-6, -30, -12]
w = 0.02 * e * x3
assert np.allclose(w, [0.12, 0.6, 0.24])
assert np.isclose(x3 @ w, 3.6)
assert np.isclose(y3 - x3 @ w, 6 * (1 - 0.02 * (x3 @ x3))) and x3 @ x3 == 30

# persamaan normal, residu, JKG, s2
wf = [Fr(4), Fr(1, 2), Fr(-1, 2)]
XtXi = [[int(v) for v in r] for r in XtX]
assert [sum(a * b for a, b in zip(r, wf)) for r in XtXi] == [25, 115, 99]
w = np.linalg.solve(XtX, Xty)
assert np.allclose(w, [4, 0.5, -0.5])
yh = X @ w
r = target - yh
assert np.allclose(yh, [5.5, 4.5, 3.5, 3, 4.5, 4])
assert np.allclose(r, [0.5, 0.5, -0.5, 1, -0.5, -1])
assert np.isclose(r @ r, 3) and np.isclose(r @ r / 3, 1)
assert np.allclose(X.T @ r, 0)
assert np.allclose(F[:, 0] * r, [2.5, 3, -2.5, 3, -2, -4])

# ramalan rekursif
f = ramal(y, w, 3)
assert np.allclose(f, [3.5, 4.25, 4.375])

# harapan AR(2)
assert np.isclose(w[0] / (1 - w[1] - w[2]), 4)
assert np.isclose(ramal(y, w, 200)[-1], 4)

# Latihan 1: JKG awal
assert target @ target == 111
print("Contoh Soal Bab 1: semua bilangan cocok")
