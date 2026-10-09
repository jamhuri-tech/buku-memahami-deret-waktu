"""Memeriksa setiap bilangan Contoh Soal dan hitungan teks Bab 15."""
import numpy as np
from scipy import stats

r24 = np.array([0.04, 0.37, 0.52, 0.25, -0.03, -0.08, -0.18, -0.03,
                -0.12, 0.08, 0.30, 0.44])
r25 = np.array([-0.76, -0.48, 1.65, 1.17, -0.37, 0.19, 0.30, -0.08,
                0.21, 0.28, 0.17, 0.64])
assert np.isclose(r24.sum(), 1.56) and np.isclose(r25.sum(), 2.92)
assert round(np.prod(1 + r24 / 100), 6) == 1.015681
assert round((np.prod(1 + r24 / 100) - 1) * 100, 3) == 1.568
assert round((np.prod(1 + r25 / 100) - 1) * 100, 3) == 2.934

c, phi, Phi = 0.1423, 0.2020, 0.2859
assert round(-phi * Phi, 4) == -0.0578
assert round((1 - phi) * (1 - Phi), 4) == 0.5699
assert round(c / ((1 - phi) * (1 - Phi)), 4) == 0.2497

assert round(28 / 33 * 100, 1) == 84.8 and np.isclose(33 * 0.95, 31.35)
p = [round(stats.binom.pmf(k, 33, 0.95), 4) for k in range(29, 34)]
assert p == [0.0578, 0.1464, 0.2692, 0.3196, 0.1840]
assert round(sum(p), 4) == 0.9770
assert round(stats.binom.cdf(28, 33, 0.95), 4) == 0.0230

c, phi, Phi, s2 = 0.1562, 0.1096, 0.2845, 0.0804
suku = [phi * 0.30, Phi * 0.28, phi * Phi * 0.21]
assert [round(v, 5) for v in suku] == [0.03288, 0.07966, 0.00655]
f = c + suku[0] + suku[1] - suku[2]
assert round(f, 4) == 0.2622
s = np.sqrt(s2)
assert round(s, 4) == 0.2835 and round(1.96 * 0.2835, 4) == 0.5557
assert (round(f - 1.96 * s, 3), round(f + 1.96 * s, 3)) == (-0.294, 0.818)
from bab15_studikasus import muat
tahun, bulan, y = muat()
latih = (tahun >= 2015) & (tahun <= 2023)
rata = [round(y[latih & (bulan == b)].mean(), 3) for b in range(1, 13)]
assert rata == [0.414, 0.043, 0.186, 0.226, 0.322, 0.422, 0.362, 0.013,
                0.124, 0.069, 0.268, 0.571]
print("Contoh Soal Bab 15: semua bilangan cocok")
