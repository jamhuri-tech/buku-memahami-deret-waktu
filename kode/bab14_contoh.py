"""Memeriksa setiap bilangan Contoh Soal dan hitungan teks Bab 14."""
from fractions import Fraction as Fr

import numpy as np

y = np.array([2.0, 5, 6, 5])
x = np.array([1.0, 2, 3, 4])
assert list(y[1:] - 0.5 * y[:-1]) == [4, 3.5, 2]
assert list(x[1:] - 0.5 * x[:-1]) == [1.5, 2, 2.5]

A = np.array([[0.5, 0.2], [0, 0.4]])
mu2 = Fr(1) / (1 - Fr(2, 5))
mu1 = (1 + Fr(1, 5) * mu2) / (1 - Fr(1, 2))
assert (mu1, mu2) == (Fr(8, 3), Fr(5, 3))
assert np.allclose(np.linalg.solve(np.eye(2) - A, [1, 1]), [8 / 3, 5 / 3])
assert np.allclose(np.array([1, 1]) + A @ [2, 2], [2.4, 1.8])

assert np.isclose(0.1 / (1 - 0.9), 1)
s1 = 0.1 + 0.1 * 4 + 0.8 * 1
assert np.isclose(s1, 1.3) and np.isclose(1 + 0.9 * 0.3, 1.27)
print("Contoh Soal Bab 14: semua bilangan cocok")
