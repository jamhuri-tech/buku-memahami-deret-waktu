"""Bab 13: ukuran galat ramalan, asal bergulir, uji Diebold-Mariano.

Fungsi yang dipakai bab-bab berikutnya:
  ukuran(e, y, skala)    MAE, RMSE, MAPE (%), MASE
  uji_dm(e1, e2, h)      statistik DM dengan koreksi HLN, nilai-p t
"""
import numpy as np
from scipy import stats


def ukuran(e, y, skala):
    """MAE, RMSE, MAPE dalam persen, dan MASE dengan pembagi skala."""
    e, y = np.asarray(e, dtype=float), np.asarray(y, dtype=float)
    mae = np.mean(np.abs(e))
    return mae, np.sqrt(np.mean(e**2)), 100 * np.mean(np.abs(e / y)), \
        mae / skala


def uji_dm(e1, e2, h=1):
    """Uji Diebold-Mariano galat kuadrat, koreksi Harvey dkk. (1997)."""
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    n = len(d)
    dc = d - d.mean()
    v = dc @ dc / n
    for k in range(1, h):
        v += 2 * (dc[:-k] @ dc[k:]) / n
    dm = d.mean() / np.sqrt(v / n)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return dm, hln, 2 * stats.t.sf(abs(hln), n - 1)


if __name__ == "__main__":
    from bab01_data import deret_mini

    y = deret_mini()
    asal = range(4, 8)                        # titik asal T = 4..7
    nyata = y[4:]
    rata = np.array([y[:T].mean() for T in asal])
    naif = np.array([y[T - 1] for T in asal])
    e_rata, e_naif = nyata - rata, nyata - naif
    skala = np.mean(np.abs(np.diff(y)))
    print("(1) asal bergulir T = 4..7, ramalan satu langkah t = 5..8:")
    print("    galat rata-rata:", " ".join(f"{v:7.4f}" for v in e_rata))
    print("    galat naif     :", " ".join(f"{v:7.4f}" for v in e_naif))
    print("             MAE    RMSE    MAPE    MASE")
    for nama, e in (("rata-rata", e_rata), ("naif     ", e_naif)):
        print(f"    {nama}", " ".join(f"{v:7.4f}"
                                    for v in ukuran(e, nyata, skala)))
    dm, hln, p = uji_dm(e_rata, e_naif)
    print(f"(2) Diebold-Mariano rata-rata lawan naif, n = 4:")
    print(f"    DM = {dm:.4f}, HLN = {hln:.4f}, p (t, db 3) = {p:.4f}")
