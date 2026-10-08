"""Bab 3: musiman multiplikatif dan transformasi log (simulasi)."""
import numpy as np
from statsmodels.tsa.seasonal import STL

from bab01_data import BENIH

rng = np.random.default_rng(BENIH)
T = 40
t = np.arange(1, T + 1)
S = np.array([0.7, 1.0, 1.4, 0.9])[(t - 1) % 4]
y = 50 * 1.04**t * S * np.exp(0.02 * rng.standard_normal(T))


def ayunan(v):
    """Selisih terbesar dan terkecil dalam setiap tahun."""
    return np.ptp(v.reshape(-1, 4), axis=1)


print("(3) musiman multiplikatif, T = 40 kuartal:")
a, b = ayunan(y), ayunan(np.log(y))
print("    rata-rata ayunan tahun 1-3 dan tahun 8-10:")
print(f"    y     : {a[:3].mean():7.2f} {a[-3:].mean():7.2f}")
print(f"    log y : {b[:3].mean():7.4f} {b[-3:].mean():7.4f}")
print(f"    log 1.4 - log 0.7 = {np.log(2):.4f}")

st = STL(np.log(y), period=4, robust=True).fit()
print("(4) STL pada log y, faktor musiman exp(S), tahun terakhir:")
print("    ", np.round(np.exp(st.seasonal[-4:]), 3))
