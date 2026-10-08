"""Bab 4: AR(1), akar karakteristik AR(2), ACF teoretis, Yule-Walker.

Fungsi yang dipakai bab-bab berikutnya:
  acf_ar(phi, H)     ACF teoretis AR(p) dari rekursi Yule-Walker
  yule_walker(y, p)  taksiran (phi, sigma2) dengan c(h) berpembagi T
"""
import numpy as np


def acf_ar(phi, H):
    """rho(0..H) untuk AR(p): sistem Yule-Walker lalu rekursi."""
    phi = np.asarray(phi, dtype=float)
    p = len(phi)
    # rho(h) = sum_j phi_j rho(|h - j|), h = 1..p, dengan rho(0) = 1
    A = np.eye(p)
    b = np.zeros(p)
    for h in range(1, p + 1):
        for j in range(1, p + 1):
            k = abs(h - j)
            if k == 0:
                b[h - 1] += phi[j - 1]
            else:
                A[h - 1, k - 1] -= phi[j - 1]
    rho = list(np.r_[1.0, np.linalg.solve(A, b)])
    for h in range(p + 1, H + 1):
        # rho(h) = phi_1 rho(h-1) + ... + phi_p rho(h-p)
        rho.append(sum(phi[j - 1] * rho[h - j] for j in range(1, p + 1)))
    return np.array(rho[:H + 1])


def yule_walker(y, p):
    """Solusi Gamma_p phi = gamma_p dengan autokovarians sampel."""
    from bab02_acf import autokov
    c = autokov(y, p)
    G = np.array([[c[abs(i - j)] for j in range(p)] for i in range(p)])
    phi = np.linalg.solve(G, c[1:])
    return phi, c[0] - phi @ c[1:]


if __name__ == "__main__":
    from statsmodels.regression.linear_model import yule_walker as yw_sm
    from statsmodels.tsa.arima_process import ArmaProcess

    from bab01_data import deret_mini, kuadrat_terkecil
    from bab02_acf import autokov

    np.set_printoptions(suppress=True)

    phi = np.array([0.5, -0.5])
    akar = np.roots([-phi[1], -phi[0], 1])
    th = np.arccos(phi[0] / (2 * np.sqrt(-phi[1])))
    print("(1) AR(2) data mini, phi = (0.5, -0.5):")
    print("    akar phi(z):", np.round(akar, 4))
    print("    |z| =", np.round(np.abs(akar), 4))
    print(f"    theta = {th:.4f} rad, periode 2 pi/theta = {2 * np.pi / th:.4f}")
    print(f"    faktor redam sqrt(-phi2) = {np.sqrt(-phi[1]):.4f}")
    rho = acf_ar(phi, 7)
    sm = ArmaProcess([1, *-phi]).acf(8)
    print("    rho(0..7):")
    print("    ", " ".join(f"{np.round(v, 10) + 0.0:6.3f}" for v in rho))
    print("    statsmodels ArmaProcess:")
    print("    ", " ".join(f"{np.round(v, 10) + 0.0:6.3f}" for v in sm))
    print(f"    gamma(0)/sigma2 = {1 / (1 - phi @ rho[1:3]):.4f}")

    y = deret_mini()
    c = autokov(y, 2)
    ph, s2 = yule_walker(y, 2)
    sm, sig = yw_sm(y, order=2, method="mle")
    w = kuadrat_terkecil(y, 2)
    print("(2) Yule-Walker pada data mini:")
    print("    c(0..2) =", c)
    print("    phi =", np.round(ph, 4), f" sigma2 = {s2:.4f}")
    print("    statsmodels:", np.round(sm, 4),
          f" sigma2 = {sig**2:.4f}")
    print("    kuadrat terkecil (Bab 1):", np.round(w[1:], 4))
