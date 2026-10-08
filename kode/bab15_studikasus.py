"""Bab 15: studi kasus inflasi bulanan Indonesia (BPS), 2006-2026.

Data: data/inflasi_bps.csv, inflasi bulan ke bulan (persen) dari tabel
BPS "Tingkat Inflasi Harga Konsumen Nasional Bulanan (M-to-M)
(2022=100)"; sumber dan lisensi di data/SUMBER.md.

Urutan keluaran mengikuti protokol bab: (1) eksplorasi dan uji akar
unit, (2) pemilihan orde dengan AICc, (3) taksiran dan diagnostik
model terpilih, (4) evaluasi asal bergulir, (5) model akhir dan
ramalan dua belas bulan.
"""
import warnings
from pathlib import Path

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

from bab13_evaluasi import uji_dm

warnings.filterwarnings("ignore")
DATA = Path(__file__).resolve().parent.parent / "data" / "inflasi_bps.csv"
LATIH = (2015, 2023)            # data latih: Januari 2015 - Desember 2023


def muat():
    """Kolom tahun, bulan, inflasi dari CSV."""
    a = np.loadtxt(DATA, delimiter=",", skiprows=1)
    return a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2]


def sarima(y, orde=(1, 0, 0), musim=(1, 0, 0, 12), exog=None):
    return SARIMAX(y, exog=exog, order=orde, seasonal_order=musim,
                   trend="c").fit(disp=False, maxiter=500)


def ramal_bulanan(y, bulan):
    """Rata-rata bulan kalender yang sama pada data y."""
    return np.array([y[bulan[:len(y)] == b].mean() for b in range(1, 13)])


if __name__ == "__main__":
    tahun, bulan, y = muat()

    print("(1) ringkasan per periode dan uji akar unit:")
    for a, b in ((2006, 2014), (2015, 2023), (2024, 2026)):
        s = y[(tahun >= a) & (tahun <= b)]
        print(f"    {a}-{b}: n = {len(s):3d}, rata-rata = {s.mean():.3f},"
              f" sb = {s.std(ddof=1):.3f}")
    for a in (2006, 2015):
        s = y[(tahun >= a) & (tahun <= 2023)]
        p_adf = adfuller(s, regression="c", autolag="AIC")[1]
        st_kpss = kpss(s, regression="c", nlags="auto")[0]
        print(f"    {a}-2023: ADF nilai-p = {p_adf:.4f},"
              f" KPSS stat = {st_kpss:.3f}")

    latih = (tahun >= LATIH[0]) & (tahun <= LATIH[1])
    s = y[latih]
    print("(2) lima model teratas menurut AICc, data latih (n = 108):")
    hasil = []
    for p in range(3):
        for q in range(3):
            for P in range(2):
                for Q in range(2):
                    m = sarima(s, (p, 0, q), (P, 0, Q, 12))
                    hasil.append((m.aicc, m.bic, p, q, P, Q))
    print("    (p,q)(P,Q)    AICc     BIC")
    for aicc, bic, p, q, P, Q in sorted(hasil)[:5]:
        print(f"    ({p},{q})({P},{Q})  {aicc:6.1f}  {bic:6.1f}")
    m22 = sarima(s, (2, 0, 2), (0, 0, 0, 0))
    print("    ARMA(2,2): |akar AR|",
          " ".join(f"{v:.3f}" for v in np.abs(m22.arroots)),
          " |akar MA|",
          " ".join(f"{v:.3f}" for v in np.abs(m22.maroots)))

    m = sarima(s)
    print("(3) SARIMA(1,0,0)(1,0,0)_12 dengan konstanta, data latih:")
    for nama, b, g in zip(("c", "phi1", "Phi1", "sigma2"),
                          m.params, m.bse):
        print(f"    {nama:6s} = {b:7.4f}   (gb {g:.4f})")
    print(f"    AICc = {m.aicc:.1f}, BIC = {m.bic:.1f}")
    lb = acorr_ljungbox(m.resid, lags=[12, 24], model_df=2)
    print(f"    Ljung-Box Q(12) = {lb.lb_stat.iloc[0]:.2f},"
          f" nilai-p = {lb.lb_pvalue.iloc[0]:.3f}")
    print(f"    Ljung-Box Q(24) = {lb.lb_stat.iloc[1]:.2f},"
          f" nilai-p = {lb.lb_pvalue.iloc[1]:.3f}")

    awal, n0 = int((tahun < LATIH[0]).sum()), int(latih.sum())
    yy, bb = y[awal:], bulan[awal:]
    skala = np.mean(np.abs(yy[12:n0] - yy[:n0 - 12]))
    e = {"naif musiman": [], "rata bulanan": [], "SARIMA": [],
         "ARMA(2,2)": []}
    cakup = 0
    for o in range(n0, len(yy)):
        tr = yy[:o]
        e["naif musiman"].append(yy[o] - tr[-12])
        e["rata bulanan"].append(yy[o] - ramal_bulanan(tr, bb)[bb[o] - 1])
        f = sarima(tr).get_forecast(1)
        e["SARIMA"].append(yy[o] - f.predicted_mean[0])
        lo, hi = f.conf_int(alpha=0.05)[0]
        cakup += lo <= yy[o] <= hi
        f = sarima(tr, (2, 0, 2), (0, 0, 0, 0)).forecast(1)
        e["ARMA(2,2)"].append(yy[o] - f[0])
    e = {k: np.array(v) for k, v in e.items()}
    bukan25 = tahun[awal + n0:] != 2025
    print(f"(4) asal bergulir h = 1, Jan 2024 - Sep 2026"
          f" (n = {len(yy) - n0}):")
    print(f"    pembagi MASE (naif musiman, data latih) = {skala:.4f}")
    print("    model            MAE    RMSE    MASE  MAE tnp 2025")
    for k, v in e.items():
        mae = np.mean(np.abs(v))
        print(f"    {k:13s}  {mae:.3f}  {np.sqrt(np.mean(v**2)):.3f}"
              f"  {mae / skala:.3f}  {np.mean(np.abs(v[bukan25])):.3f}")
    for a, b in (("SARIMA", "naif musiman"), ("SARIMA", "rata bulanan"),
                 ("ARMA(2,2)", "SARIMA")):
        _, hln, p = uji_dm(e[a], e[b], 1)
        print(f"    DM {a} vs {b}: HLN {hln:.3f}, p {p:.3f}")
    print(f"    selang 95% SARIMA memuat nilai nyata: {cakup} dari"
          f" {len(yy) - n0}")

    d = np.zeros((len(yy), 3))
    for j in range(3):
        d[(tahun[awal:] == 2025) & (bb == j + 1), j] = 1
    m = sarima(yy, exog=d)
    print("(5) model akhir, Jan 2015 - Sep 2026, dummy Jan-Mar 2025:")
    for nama, b, g in zip(("c", "d_jan25", "d_feb25", "d_mar25", "phi1",
                           "Phi1", "sigma2"), m.params, m.bse):
        print(f"    {nama:7s} = {b:7.4f}   (gb {g:.4f})")
    lb = acorr_ljungbox(m.resid, lags=[12, 24], model_df=2)
    print(f"    Ljung-Box nilai-p: Q(12) {lb.lb_pvalue.iloc[0]:.3f},"
          f" Q(24) {lb.lb_pvalue.iloc[1]:.3f}")
    f = m.get_forecast(12, exog=np.zeros((12, 3)))
    sel = f.conf_int(alpha=0.05)
    nama = ("Okt", "Nov", "Des", "Jan", "Feb", "Mar", "Apr", "Mei",
            "Jun", "Jul", "Agu", "Sep")
    print("    bulan   ramalan  selang 95%")
    for k in range(12):
        print(f"    {nama[k]:3s}     {f.predicted_mean[k]:6.3f}"
              f"  [{sel[k, 0]:6.3f}, {sel[k, 1]:6.3f}]")
    tot = (np.prod(1 + f.predicted_mean / 100) - 1) * 100
    print(f"    jumlah 12 ramalan = {f.predicted_mean.sum():.3f},"
          f" majemuk = {tot:.3f}")
