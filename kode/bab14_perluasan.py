"""Bab 14: transformasi Cochrane-Orcutt, VAR(1), ramalan varians GARCH.

Fungsi yang dipakai di luar bab ini:
  garch_varians(e, omega, alpha, beta, s2_0)  rekursi sigma_t^2
"""
import numpy as np


def garch_varians(e, omega, alpha, beta, s2_0):
    """sigma2_t = omega + alpha e_{t-1}^2 + beta sigma2_{t-1}."""
    s2 = np.empty(len(e))
    s2[0] = s2_0
    for t in range(1, len(e)):
        s2[t] = omega + alpha * e[t - 1] ** 2 + beta * s2[t - 1]
    return s2


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    y = np.array([2.0, 5, 6, 5])
    x = np.array([1.0, 2, 3, 4])
    phi = 0.5
    print("(1) Cochrane-Orcutt, phi = 0.5:")
    print("    y* =", y[1:] - phi * y[:-1], " x* =", x[1:] - phi * x[:-1])

    A = np.array([[0.5, 0.2], [0.0, 0.4]])
    c = np.array([1.0, 1.0])
    mu = np.linalg.solve(np.eye(2) - A, c)
    print("(2) VAR(1):")
    print("    nilai eigen A =", np.round(np.linalg.eigvals(A), 4),
          " mu =", np.round(mu, 4))
    print("    ramalan dari y_T = (2, 2):", c + A @ np.array([2.0, 2.0]))

    om, al, be = 0.1, 0.1, 0.8
    v = om / (1 - al - be)
    s1 = om + al * 2.0**2 + be * 1.0
    print("(3) GARCH(1,1), omega = 0.1, alpha = 0.1, beta = 0.8:")
    print(f"    varians takbersyarat = {v:.4f}, sigma2_(t+1) = {s1:.4f}")
    print("    sigma2, h = 1..5:",
          " ".join(f"{v + (al + be) ** (h - 1) * (s1 - v):.4f}"
                   for h in range(1, 6)))
