"""Bab 6: MA(q), ARMA(p, q), bobot psi dan pi, keterbalikan.

Fungsi yang dipakai bab-bab berikutnya:
  bobot_psi(phi, theta, J)  psi_0..psi_J dari phi(z) psi(z) = theta(z)
  bobot_pi(phi, theta, J)   pi_0..pi_J dari theta(z) pi(z) = phi(z)
  acf_arma(phi, theta, H)   rho(0..H) dari bobot psi (dipotong panjang)
"""
import numpy as np


def bobot_psi(phi, theta, J):
    """psi_j = theta_j + sum_k phi_k psi_{j-k}, psi_0 = 1."""
    phi, theta = list(phi), list(theta)
    psi = [1.0]
    for j in range(1, J + 1):
        v = theta[j - 1] if j <= len(theta) else 0.0
        v += sum(phi[k - 1] * psi[j - k] for k in range(1, len(phi) + 1)
                 if j - k >= 0)
        psi.append(v)
    return np.array(psi)


def bobot_pi(phi, theta, J):
    """Bobot AR tak hingga: theta(z) pi(z) = phi(z), pi_0 = 1.

    Model ditulis eps_t = sum_j pi_j (y_{t-j} - mu), sehingga
    pi_j = -phi_j - sum_k theta_k pi_{j-k}.
    """
    phi, theta = list(phi), list(theta)
    pi = [1.0]
    for j in range(1, J + 1):
        v = -phi[j - 1] if j <= len(phi) else 0.0
        v -= sum(theta[k - 1] * pi[j - k] for k in range(1, len(theta) + 1)
                 if j - k >= 0)
        pi.append(v)
    return np.array(pi)


def acf_arma(phi, theta, H, J=2000):
    """rho(0..H) dari gamma(h) = sigma2 sum_j psi_j psi_{j+h}."""
    psi = bobot_psi(phi, theta, J + H)
    g = np.array([psi[:J] @ psi[h:J + h] for h in range(H + 1)])
    return g / g[0]


if __name__ == "__main__":
    from statsmodels.tsa.arima_process import ArmaProcess

    from bab01_data import deret_mini
    from bab02_acf import acf_sampel

    np.set_printoptions(suppress=True)

    def baris(v):
        return " ".join(f"{np.round(x, 10) + 0.0:7.4f}" for x in v)

    print("(1) MA(1): theta dan 1/theta memberi ACF yang sama")
    for th in (0.5, 2.0):
        print(f"    theta = {th}: rho(1..3) =",
              baris(ArmaProcess([1], [1, th]).acf(4)[1:]))
    r1 = acf_sampel(deret_mini(), 1)[1]
    akar = np.roots([r1, -1, r1])
    print(f"    data mini r(1) = {r1:.4f}")
    print("    akar theta/(1 + theta^2) = r(1):", np.round(np.sort(akar), 4))

    print("(2) ACF dari bobot psi dan dari statsmodels, h = 1..4:")
    for nama, phi, theta in [("MA(2) (1, 0.5)", [], [1, 0.5]),
                             ("ARMA(1,1) 0.5/0.5", [0.5], [0.5]),
                             ("ARMA(1,1) 0.5/-0.5", [0.5], [-0.5])]:
        sm = ArmaProcess(np.r_[1, -np.array(phi)], np.r_[1, theta])
        print(f"    {nama}")
        print("      psi_1..4 :", baris(bobot_psi(phi, theta, 4)[1:]))
        print("      rho kita :", baris(acf_arma(phi, theta, 4)[1:]))
        print("      statsm.  :", baris(sm.acf(5)[1:]))
