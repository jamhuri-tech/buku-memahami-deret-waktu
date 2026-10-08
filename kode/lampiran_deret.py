"""Lampiran B: inti buku dalam satu berkas (NumPy dan SciPy).

ACF, Ljung-Box, PACF (Durbin-Levinson), AR(p) dari tabel lag,
bobot psi, ramalan dengan selang 95%, dan AIC/AICc/BIC.
"""
import numpy as np
from scipy import stats


def acf(y, H):
    d = y - y.mean()
    c = [d[:len(y) - h] @ d[h:] for h in range(H + 1)]
    return np.array(c) / (d @ d)


def ljung_box(y, H, k=0):
    T, r, h = len(y), acf(y, H)[1:], np.arange(1, H + 1)
    Q = T * (T + 2) * np.sum(r**2 / (T - h))
    return Q, stats.chi2.sf(Q, H - k)


def pacf(y, H):
    rho, alfa, phi = acf(y, H), [], np.zeros(0)
    for h in range(1, H + 1):
        a = rho[h] - phi @ rho[h - 1:0:-1]
        a /= 1 - phi @ rho[1:h]
        phi = np.r_[phi - a * phi[::-1], a]
        alfa.append(a)
    return np.array(alfa)


def ar_kt(y, p):
    """w0..wp, s^2, dan residu dari tabel lag t = p+1..T."""
    T = len(y)
    kol = [y[p - j:T - j] for j in range(1, p + 1)]
    X = np.column_stack([np.ones(T - p)] + kol)
    w = np.linalg.solve(X.T @ X, X.T @ y[p:])
    e = y[p:] - X @ w
    return w, e @ e / (T - 2 * p - 1), e


def psi(phi, h):
    s = [1.0]
    for j in range(1, h):
        k = range(min(j, len(phi)))
        s.append(sum(phi[i] * s[j - i - 1] for i in k))
    return np.array(s)


def ramal(y, w, s2, h):
    r, p = list(y), len(w) - 1
    for _ in range(h):
        r.append(w[0] + w[1:] @ np.array(r[::-1][:p]))
    f = np.array(r[len(y):])
    gb = np.sqrt(s2 * np.cumsum(psi(w[1:], h) ** 2))
    return f, f - 1.96 * gb, f + 1.96 * gb


def kriteria(jkg, m, k):
    dasar = m * np.log(jkg / m)
    aicc = dasar + 2 * k + 2 * k * (k + 1) / (m - k - 1)
    return dasar + 2 * k, aicc, dasar + k * np.log(m)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    y = np.array([2.0, 5, 6, 5, 3, 4, 4, 3])
    print("ACF r(1..3):", acf(y, 3)[1:])
    print("Ljung-Box H = 2: Q = %.4f, p = %.4f"
          % ljung_box(y, 2))
    print("PACF a(1..3):", pacf(y, 3))
    w, s2, e = ar_kt(y, 2)
    print("AR(2): w =", w, " s2 =", s2, " JKG =", e @ e)
    print("psi(0..3):", psi(w[1:], 4))
    f, lo, hi = ramal(y, w, s2, 3)
    print("ramalan t = 9..11:", f)
    print("batas bawah 95%:", lo)
    print("batas atas 95% :", hi)
    print("AIC, AICc, BIC:", np.array(kriteria(e @ e, 6, 3)))
