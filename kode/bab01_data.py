"""Data mini yang dipakai di seluruh buku, dan tabel lag-nya.

Penjualan mingguan sebuah warung (ratusan unit) selama delapan pekan,
t = 1, ..., 8. Rata-ratanya 4. Tabel lag AR(2) memuat baris
t = 3, ..., 8 dengan fitur x1 = y(t-1), x2 = y(t-2) dan target y(t);
kuadrat terkecilnya tepat w = (4, 1/2, -1/2) dengan JKG 3.
"""
import numpy as np

BENIH = 20261008

Y_MINI = np.array([2, 5, 6, 5, 3, 4, 4, 3], dtype=float)


def deret_mini():
    """Deret y_1, ..., y_8 (indeks NumPy 0, ..., 7)."""
    return Y_MINI.copy()


def tabel_lag(y, p):
    """Tabel lag orde p: baris t = p+1, ..., T.

    Mengembalikan (t, F, target): nomor waktu (mulai 1), matriks fitur
    berukuran (T - p) x p dengan kolom j berisi y(t-j), dan y(t).
    """
    y = np.asarray(y, dtype=float)
    T = len(y)
    t = np.arange(p + 1, T + 1)
    F = np.column_stack([y[t - 1 - j] for j in range(1, p + 1)])
    return t, F, y[t - 1]


def rancang(F):
    """Menambahkan kolom x0 = 1: X berukuran m x (n + 1)."""
    F = np.asarray(F, dtype=float)
    return np.column_stack([np.ones(len(F)), F])


def kuadrat_terkecil(y, p):
    """Bobot (w0, ..., wp) dari persamaan normal X'X w = X'y."""
    _, F, target = tabel_lag(y, p)
    X = rancang(F)
    return np.linalg.solve(X.T @ X, X.T @ target)


def ramal(y, w, h):
    """Ramalan rekursif h langkah ke depan dengan bobot AR w."""
    riwayat = list(np.asarray(y, dtype=float))
    p = len(w) - 1
    hasil = []
    for _ in range(h):
        lag = riwayat[::-1][:p]
        yh = w[0] + float(np.dot(w[1:], lag))
        hasil.append(yh)
        riwayat.append(yh)
    return np.array(hasil)


if __name__ == "__main__":
    y = deret_mini()
    print("y:", y, "rata-rata:", y.mean())
    print("w:", kuadrat_terkecil(y, 2))
