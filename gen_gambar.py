# -*- coding: utf-8 -*-
"""Membangkitkan seluruh gambar Matplotlib ke gbr/ dalam dua bentuk:
PDF vektor untuk cetak dan PNG 300 dpi untuk EPUB.

Satu fungsi per gambar, dinamai babNN_nama(), yang memanggil
simpan(fig, "babNN-nama"). Fungsi bernama babNN_* dijalankan otomatis.
Benih acak selalu tetap, supaya gambar tidak berubah setiap build.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENIH = 20261008  # sama dengan seluruh kode/bab*.py
GBR = Path("gbr")

# Sebagian gambar memakai kelas yang sudah ditulis di kode/, supaya
# logikanya tidak terduplikasi di dua tempat.
sys.path.insert(0, str(Path(__file__).parent / "kode"))

# Warna mengikuti preamble.tex.
BIRU = "#1B3B6F"
HIJAU = "#1E6F5C"
JINGGA = "#B85C00"
MERAH = "#9B1B30"
ABU = "#5A6472"
ABU_GARIS = "#C9CED6"
BIRU_MUDA = "#E8EEF7"
HIJAU_MUDA = "#E6F2EF"
JINGGA_MUDA = "#FDF0E3"
MERAH_MUDA = "#FBE9EC"

plt.rcParams.update({
    "font.size": 7.5,
    "axes.edgecolor": ABU_GARIS,
    "axes.labelcolor": ABU,
    "axes.titlesize": 8,
    "axes.titlecolor": BIRU,
    "xtick.color": ABU,
    "ytick.color": ABU,
    "text.color": ABU,
    "grid.color": ABU_GARIS,
    "legend.frameon": False,
    "figure.dpi": 300,
})


def angka(v, n=3):
    """Angka dengan koma desimal, sesuai kaidah bahasa Indonesia."""
    return f"{v:.{n}f}".replace(".", ",")


def angka_mat(v, n=3):
    """Seperti angka(), untuk mode matematika: koma tanpa spasi."""
    return angka(v, n).replace(",", "{,}")


def _koma(fig):
    """Mengubah pemisah desimal pada label sumbu menjadi koma.

    Sumbu berskala logaritmik dilewati, karena labelnya berupa pangkat
    sepuluh dan tidak memuat pemisah desimal.
    """
    from matplotlib.ticker import FuncFormatter
    rapi = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for ax in fig.axes:
        kunci = getattr(ax, "_label_terkunci", set())
        if "x" not in kunci and ax.get_xscale() == "linear":
            ax.xaxis.set_major_formatter(rapi)
        if "y" not in kunci and ax.get_yscale() == "linear":
            ax.yaxis.set_major_formatter(rapi)


def kunci_label(ax, *sumbu):
    """Menandai sumbu yang labelnya kita tetapkan sendiri."""
    ax._label_terkunci = getattr(ax, "_label_terkunci",
                                 set()) | set(sumbu)


def simpan(fig, nama):
    _koma(fig)
    GBR.mkdir(exist_ok=True)
    fig.savefig(GBR / f"{nama}.pdf", bbox_inches="tight")
    fig.savefig(GBR / f"{nama}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("gbr/" + nama)


def _rapikan(ax):
    """Gaya sumbu seri: tanpa bingkai atas dan kanan."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------
#  Gambar per bab ditambahkan di bawah ini sebagai fungsi babNN_nama().





# ============================ Bab 1 ==================================

# ============================ Bab 1 ==================================

def bab01_deret():
    from bab01_data import deret_mini
    y = deret_mini()
    t = np.arange(1, 9)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.0))
    a.axhline(4, color=ABU_GARIS, lw=0.8, ls="--")
    a.plot(t, y, color=BIRU, lw=1.0, marker="o", ms=3.5)
    for ti, yi in zip(t, y):
        a.annotate(f"{yi:.0f}", (ti, yi + 0.25), ha="center", fontsize=6)
    a.set_ylim(0, 7.2)
    a.set_xlabel("pekan $t$")
    a.set_ylabel("$y_t$ (ratus unit)")
    a.set_title(r"deret $y_t$, rata-rata 4")
    d = np.diff(y)
    b.axhline(0, color=ABU_GARIS, lw=0.8)
    b.bar(t[1:], d, color=np.where(d >= 0, BIRU, JINGGA), width=0.55)
    for ti, di in zip(t[1:], d):
        b.annotate(f"{di:.0f}".replace("-", "\u2212"),
                   (ti, di + (0.25 if di >= 0 else -0.6)),
                   ha="center", fontsize=6)
    b.set_xlim(0.5, 8.5)
    b.set_ylim(-3, 4)
    b.set_xlabel("pekan $t$")
    b.set_ylabel(r"$\nabla y_t = y_t - y_{t-1}$")
    b.set_title("diferensi pertama")
    for ax in (a, b):
        ax.set_xticks(t)
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab01-deret")


def bab01_arsitektur():
    from matplotlib.patches import Circle, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(4.8, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def simpul(x, y, teks, r=0.42, wr=BIRU, isi=BIRU_MUDA):
        ax.add_patch(Circle((x, y), r, facecolor=isi, edgecolor=wr,
                            lw=0.8))
        ax.text(x, y, teks, ha="center", va="center", fontsize=6.5)

    def panah(a, b, wr=ABU, teks=None, pos=0.5, dy=0.15, ls="-"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>",
                                     mutation_scale=7, color=wr, lw=0.7,
                                     linestyle=ls))
        if teks:
            x = a[0] + pos * (b[0] - a[0])
            y = a[1] + pos * (b[1] - a[1]) + dy
            ax.text(x, y, teks, ha="center", fontsize=6, color=wr)

    masuk = [(0.8, 4.3, "$1$"), (0.8, 3.2, "$y_{t-1}$"),
             (0.8, 1.6, "$y_{t-p}$")]
    ax.text(0.8, 2.45, r"$\vdots$", ha="center", fontsize=8)
    for x, yy, tk in masuk:
        simpul(x, yy, tk)
    simpul(3.6, 2.9, r"$\Sigma$")
    for (x, yy, tk), wt in zip(masuk, ["$w_0$", "$w_1$", "$w_p$"]):
        panah((x + 0.45, yy), (3.18, 2.9), teks=wt, pos=0.45, dy=0.12)
    simpul(5.6, 2.9, r"$e_t$")
    panah((4.02, 2.9), (5.18, 2.9), teks=r"$\hat y_t$")
    ax.text(5.6, 4.3, "$y_t$", ha="center", fontsize=7)
    panah((5.6, 4.1), (5.6, 3.32))
    simpul(7.6, 2.9, r"$\ell_t$", wr=MERAH, isi=MERAH_MUDA)
    panah((6.02, 2.9), (7.18, 2.9), teks=r"$\frac{1}{2} e_t^2$")
    yb = 1.0
    ax.text(7.6, yb, r"$\dfrac{\partial\ell_t}{\partial e_t} = e_t$",
            ha="center", fontsize=6.5, color=MERAH)
    ax.text(5.6, yb, r"$\dfrac{\partial e_t}{\partial\hat y_t} = -1$",
            ha="center", fontsize=6.5, color=MERAH)
    ax.text(3.3, yb,
            r"$\dfrac{\partial\hat y_t}{\partial w_j} = x_{tj}$",
            ha="center", fontsize=6.5, color=MERAH)
    panah((7.1, 0.55), (3.0, 0.55), wr=MERAH)
    ax.text(5.3, 0.0,
            r"$\dfrac{\partial\ell_t}{\partial w_j} = -e_t\,x_{tj}$",
            ha="center", fontsize=7, color=MERAH)
    for x, tk in [(0.8, "lag (fitur)"), (3.6, "ramalan"), (5.6, "galat"),
                  (7.6, "loss")]:
        ax.text(x, 5.0, tk, ha="center", fontsize=6.5, color=ABU)
    fig.tight_layout()
    simpan(fig, "bab01-arsitektur")


def bab01_ramalan():
    from bab01_data import deret_mini, kuadrat_terkecil, ramal
    y = deret_mini()
    w = kuadrat_terkecil(y, 2)
    f = ramal(y, w, 6)
    yh = [w[0] + w[1] * y[t - 1] + w[2] * y[t - 2] for t in range(2, 8)]
    fig, ax = plt.subplots(figsize=(4.7, 2.1))
    ax.axhline(4, color=ABU_GARIS, lw=0.8, ls="--")
    ax.axvspan(8.5, 14.5, color=HIJAU_MUDA, lw=0)
    ax.plot(np.arange(1, 9), y, color=BIRU, lw=1.0, marker="o", ms=3.5,
            label="data $y_t$")
    ax.plot(np.arange(3, 9), yh, color=JINGGA, lw=0, marker="x", ms=4.5,
            label=r"ramalan satu langkah $\hat y_t$")
    for t, a, b in zip(range(3, 9), y[2:], yh):
        ax.plot([t, t], [a, b], color=JINGGA, lw=0.6, ls=":")
    ax.plot(np.r_[8, np.arange(9, 15)], np.r_[y[-1], f], color=HIJAU,
            lw=1.0, marker="s", ms=3, label="ramalan rekursif")
    ax.text(11.5, 6.2, "masa depan", ha="center", fontsize=6.5,
            color=HIJAU)
    ax.set_xticks(np.arange(1, 15))
    ax.set_ylim(1, 7)
    ax.set_xlabel("pekan $t$")
    ax.set_ylabel("$y_t$")
    ax.legend(loc="upper right", fontsize=5.5, bbox_to_anchor=(0.6, 1.0))
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab01-ramalan")


# ============================ Bab 2 ==================================

def _stem(ax, r, warna=BIRU):
    h = np.arange(len(r))
    ax.vlines(h, 0, r, color=warna, lw=1.0)
    ax.scatter(h, r, color=warna, s=8, zorder=3)
    ax.axhline(0, color=ABU, lw=0.6)


def bab02_acf():
    from bab01_data import deret_mini
    from bab02_acf import acf_sampel
    y = deret_mini()
    T = len(y)
    r = acf_sampel(y, 7)
    fig, ax = plt.subplots(figsize=(4.7, 2.0))
    b = 1.96 / np.sqrt(T)
    ax.axhspan(-b, b, color=BIRU_MUDA, lw=0)
    h = np.arange(1, 8)
    vb = [(1 + 2 * np.sum(r[1:k] ** 2)) / T for k in h]
    ax.plot(h, 1.96 * np.sqrt(vb), color=JINGGA, lw=0.8, ls="--",
            drawstyle="steps-mid", label="batas Bartlett")
    ax.plot(h, -1.96 * np.sqrt(vb), color=JINGGA, lw=0.8, ls="--",
            drawstyle="steps-mid")
    _stem(ax, r)
    teks = ["1", "1/12", "−5/12", "−1/6"]
    for k, t in enumerate(teks):
        ax.annotate(t, (k + 0.12, r[k] + (0.05 if r[k] >= 0 else -0.12)),
                    fontsize=6)
    ax.text(7.4, b - 0.12, r"$\pm 1{,}96/\sqrt{8}$", fontsize=6,
            color=BIRU, ha="right")
    ax.set_xticks(np.arange(8))
    ax.set_ylim(-1.0, 1.15)
    ax.set_xlabel("lag $h$")
    ax.set_ylabel("$r(h)$")
    ax.legend(loc="upper right", fontsize=5.5)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab02-acf")


def bab02_tiga():
    from bab02_acf import acf_sampel
    rng = np.random.default_rng(BENIH)
    T = 100
    t = np.arange(1, T + 1)
    deret = [("white noise", rng.standard_normal(T)),
             ("random walk", np.cumsum(rng.standard_normal(T))),
             ("tren + noise", 0.1 * t + rng.standard_normal(T))]
    fig, ax = plt.subplots(2, 3, figsize=(4.8, 2.9))
    b = 1.96 / np.sqrt(T)
    for k, (nama, s) in enumerate(deret):
        ax[0, k].plot(t, s, color=BIRU, lw=0.6)
        ax[0, k].set_title(nama)
        ax[0, k].set_xlabel("$t$")
        ax[1, k].axhspan(-b, b, color=BIRU_MUDA, lw=0)
        _stem(ax[1, k], acf_sampel(s, 20))
        ax[1, k].set_ylim(-0.4, 1.05)
        ax[1, k].set_xlabel("lag $h$")
    ax[0, 0].set_ylabel("$y_t$")
    ax[1, 0].set_ylabel("$r(h)$")
    for a in ax.flat:
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab02-tiga")


def bab02_simulasi():
    from scipy import stats
    from bab02_acf import acf_sampel, ljung_box
    rng = np.random.default_rng(BENIH)
    R, T, H = 2000, 100, 10
    r1 = np.empty(R)
    Q = np.empty(R)
    for k in range(R):
        e = rng.standard_normal(T)
        r1[k] = acf_sampel(e, 1)[1]
        Q[k] = ljung_box(e, H)[0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.0))
    a.hist(r1, bins=40, density=True, color=BIRU_MUDA, edgecolor=BIRU,
           lw=0.4)
    g = np.linspace(-0.4, 0.4, 200)
    a.plot(g, stats.norm.pdf(g, 0, 1 / np.sqrt(T)), color=MERAH, lw=0.9)
    a.set_xlabel("$r(1)$")
    a.set_title(r"$r(1)$ lawan $\mathcal{N}(0, 1/T)$")
    b.hist(Q, bins=40, density=True, color=BIRU_MUDA, edgecolor=BIRU,
           lw=0.4)
    g = np.linspace(0, 35, 200)
    b.plot(g, stats.chi2.pdf(g, H), color=MERAH, lw=0.9)
    b.axvline(stats.chi2.ppf(0.95, H), color=JINGGA, lw=0.8, ls="--")
    b.set_xlabel("$Q$")
    b.set_title("Ljung\u2013Box lawan "
                r"$\chi^2_{10}$")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab02-simulasi")


# ============================ Bab 3 ==================================

def bab03_data():
    from bab01_data import deret_musiman
    y = deret_musiman()
    t = np.arange(1, 13)
    kw = (t - 1) % 4
    w = np.array([1, 1, 4, 7, 1.0])
    g = np.linspace(1, 12, 200)
    fig, ax = plt.subplots(figsize=(4.7, 2.2))
    ax.plot(t, y, color=ABU_GARIS, lw=0.8, zorder=1)
    warna = [JINGGA, BIRU, MERAH, HIJAU]
    for q in range(4):
        s_ = kw == q
        ax.scatter(t[s_], y[s_], color=warna[q], s=16, zorder=3,
                   label=f"kuartal {q + 1}")
        geser = 0 if q == 0 else w[1 + q]
        ax.plot(g, w[0] + w[1] * g + geser, color=warna[q], lw=0.6,
                ls="--")
    for x in (4.5, 8.5):
        ax.axvline(x, color=ABU_GARIS, lw=0.5, ls=":")
    ax.set_xticks(t)
    ax.set_xlabel("kuartal ke-$t$")
    ax.set_ylabel("$y_t$ (ratus orang)")
    ax.legend(loc="upper left", fontsize=5.5, ncol=2)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab03-data")


def bab03_dekomposisi():
    from statsmodels.tsa.seasonal import seasonal_decompose
    from bab01_data import deret_musiman
    y = deret_musiman()
    t = np.arange(1, 13)
    dk = seasonal_decompose(y, model="additive", period=4)
    fig, ax = plt.subplots(2, 2, figsize=(4.8, 3.0))
    a = ax[0, 0]
    a.plot(t, y, color=BIRU, lw=0.9, marker="o", ms=2.5, label="$y_t$")
    a.plot(t, dk.trend, color=MERAH, lw=1.0, marker="s", ms=2.5,
           label="$m_t$ (MA $2\\times 4$)")
    a.set_title("data dan tren")
    a.legend(fontsize=5.5, loc="upper left")
    a = ax[0, 1]
    a.axhline(0, color=ABU, lw=0.5)
    a.bar(t, dk.seasonal, color=np.where(dk.seasonal >= 0, BIRU, JINGGA),
          width=0.6)
    a.set_title("musiman $S_t$")
    a = ax[1, 0]
    a.axhline(0, color=ABU, lw=0.5)
    a.bar(t, dk.resid, color=ABU, width=0.6)
    a.set_title(r"sisa $R_t = y_t - m_t - S_t$")
    a = ax[1, 1]
    d4 = y[4:] - y[:-4]
    a.axhline(4, color=MERAH, lw=0.7, ls="--")
    a.bar(t[4:], d4, color=HIJAU, width=0.6)
    a.set_xlim(0.5, 12.5)
    a.set_title(r"diferensi musiman $\nabla_4 y_t$")
    for b in ax.flat:
        b.set_xlim(0.5, 12.5)
        b.set_xticks([1, 4, 8, 12])
        b.set_xlabel("$t$")
        _rapikan(b)
    fig.tight_layout()
    simpan(fig, "bab03-dekomposisi")


def bab03_kali():
    rng = np.random.default_rng(BENIH)
    T = 40
    t = np.arange(1, T + 1)
    S = np.array([0.7, 1.0, 1.4, 0.9])[(t - 1) % 4]
    y = 50 * 1.04**t * S * np.exp(0.02 * rng.standard_normal(T))
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.0))
    a.plot(t, y, color=BIRU, lw=0.8)
    a.set_title("$y_t$: ayunan membesar")
    b.plot(t, np.log(y), color=HIJAU, lw=0.8)
    b.set_title(r"$\log y_t$: ayunan tetap")
    for ax in (a, b):
        ax.set_xlabel("$t$ (kuartal)")
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab03-kali")


# ============================ Bab 4 ==================================

def bab04_ar1():
    from statsmodels.tsa.arima_process import ArmaProcess
    rng = np.random.default_rng(BENIH)
    T = 100
    fig, ax = plt.subplots(2, 3, figsize=(4.8, 2.9))
    for k, phi in enumerate((0.9, 0.3, -0.7)):
        y = ArmaProcess([1, -phi]).generate_sample(
            T, burnin=100, distrvs=rng.standard_normal)
        ax[0, k].plot(np.arange(1, T + 1), y, color=BIRU, lw=0.6)
        ax[0, k].axhline(0, color=ABU_GARIS, lw=0.5)
        ax[0, k].set_title(f"$\\phi = {phi}$".replace(".", "{,}"))
        ax[0, k].set_xlabel("$t$")
        h = np.arange(0, 11)
        _stem(ax[1, k], phi ** h)
        ax[1, k].set_ylim(-1.05, 1.05)
        ax[1, k].set_xlabel("lag $h$")
    ax[0, 0].set_ylabel("$y_t$")
    ax[1, 0].set_ylabel(r"$\rho(h) = \phi^h$")
    for a in ax.flat:
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab04-ar1")


def bab04_segitiga():
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.fill([-2, 0, 2], [-1, 1, -1], color=BIRU_MUDA, lw=0)
    ax.plot([-2, 0, 2, -2], [-1, 1, -1, -1], color=BIRU, lw=0.8)
    g = np.linspace(-2, 2, 200)
    ax.plot(g, -g**2 / 4, color=JINGGA, lw=0.8, ls="--")
    ax.text(0, -0.72, "akar kompleks\n(gelombang teredam)", ha="center",
            fontsize=6, color=JINGGA)
    ax.text(-1.05, 0.25, "akar real", fontsize=6, color=BIRU)
    ax.scatter([0.5], [-0.5], color=MERAH, s=18, zorder=3)
    ax.annotate("data mini (KT)\n$(0{,}5;\\,-0{,}5)$", (0.5, -0.5),
                (0.95, -0.2), fontsize=5.5, color=MERAH,
                arrowprops=dict(arrowstyle="-", color=MERAH, lw=0.5))
    ax.scatter([17 / 143], [-61 / 143], color=HIJAU, s=14, marker="s",
               zorder=3)
    ax.annotate("Yule--Walker".replace("--", "–"),
                (17 / 143, -61 / 143), (-1.45, -0.62), fontsize=5.5,
                color=HIJAU,
                arrowprops=dict(arrowstyle="-", color=HIJAU, lw=0.5))
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel(r"$\phi_1$")
    ax.set_ylabel(r"$\phi_2$")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab04-segitiga")


def bab04_acf():
    from bab01_data import deret_mini
    from bab02_acf import acf_sampel
    from bab04_ar import acf_ar
    h = np.arange(0, 11)
    rho = acf_ar([0.5, -0.5], 10)
    fig, ax = plt.subplots(figsize=(4.7, 2.0))
    _stem(ax, rho)
    g = np.linspace(0, 10, 200)
    ax.plot(g, np.sqrt(0.5) ** g, color=JINGGA, lw=0.7, ls="--")
    ax.plot(g, -np.sqrt(0.5) ** g, color=JINGGA, lw=0.7, ls="--",
            label=r"$\pm(\sqrt{1/2})^h$")
    r = acf_sampel(deret_mini(), 7)
    ax.scatter(np.arange(8) + 0.15, r, color=MERAH, marker="x", s=14,
               zorder=4, label="ACF sampel data mini")
    ax.set_xticks(h)
    ax.set_xlabel("lag $h$")
    ax.set_ylabel(r"$\rho(h)$")
    ax.legend(fontsize=5.5, loc="upper right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab04-acf")


# ============================ Bab 5 ==================================

def bab05_pola():
    from statsmodels.tsa.arima_process import ArmaProcess
    proses = [("AR(1), $\\phi = 0{,}7$", [1, -0.7], [1]),
              ("AR(2), $\\phi = (\\frac{1}{2}, -\\frac{1}{2})$", [1, -0.5, 0.5],
               [1]),
              ("MA(1), $\\theta = 0{,}5$", [1], [1, 0.5])]
    fig, ax = plt.subplots(2, 3, figsize=(4.8, 2.8))
    for k, (nama, ar, ma) in enumerate(proses):
        p = ArmaProcess(ar, ma)
        _stem(ax[0, k], np.r_[np.nan, p.acf(9)[1:]])
        _stem(ax[1, k], np.r_[np.nan, p.pacf(9)[1:]], warna=HIJAU)
        ax[0, k].set_title(nama)
        for a in ax[:, k]:
            a.set_ylim(-0.75, 0.85)
            a.set_xticks([1, 4, 8])
        ax[1, k].set_xlabel("lag $h$")
    ax[0, 0].set_ylabel(r"ACF $\rho(h)$")
    ax[1, 0].set_ylabel(r"PACF $\alpha(h)$")
    for a in ax.flat:
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab05-pola")


def bab05_mini():
    from statsmodels.tsa.arima_process import ArmaProcess
    from bab01_data import deret_mini
    from bab05_pacf import pacf_sampel
    a = pacf_sampel(deret_mini(), 5)
    teori = ArmaProcess([1, -0.5, 0.5]).pacf(6)[1:]
    h = np.arange(1, 6)
    fig, ax = plt.subplots(figsize=(4.7, 1.9))
    b = 1.96 / np.sqrt(8)
    ax.axhspan(-b, b, color=BIRU_MUDA, lw=0)
    ax.axhline(0, color=ABU, lw=0.6)
    ax.vlines(h - 0.08, 0, teori, color=HIJAU, lw=1.0)
    ax.scatter(h - 0.08, teori, color=HIJAU, s=8, zorder=3,
               label=r"teoretis, $\phi = (\frac{1}{2}, -\frac{1}{2})$")
    ax.vlines(h + 0.08, 0, a, color=MERAH, lw=1.0)
    ax.scatter(h + 0.08, a, color=MERAH, marker="x", s=12, zorder=3,
               label="sampel data mini")
    ax.set_xticks(h)
    ax.set_ylim(-0.8, 0.8)
    ax.set_xlabel("lag $h$")
    ax.set_ylabel(r"$\alpha(h)$")
    ax.legend(fontsize=5.5, loc="upper right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab05-mini")


def bab05_identifikasi():
    from statsmodels.tsa.arima_process import ArmaProcess
    from bab02_acf import acf_sampel
    from bab05_pacf import pacf_sampel
    # urutan pembangkitan sama dengan kode/bab05_simulasi.py
    rng = np.random.default_rng(BENIH)
    ar2 = ArmaProcess([1, -0.5, 0.5])
    for _ in range(2000):
        ar2.generate_sample(200, burnin=200, distrvs=rng.standard_normal)
    T = 200
    b = 1.96 / np.sqrt(T)
    fig, ax = plt.subplots(2, 2, figsize=(4.7, 2.7))
    for k, (nama, proses) in enumerate(
            [("deret A", ArmaProcess([1, -0.6, 0.3])),
             ("deret B", ArmaProcess([1], [1, 0.7]))]):
        y = proses.generate_sample(T, burnin=200,
                                   distrvs=rng.standard_normal)
        for j, (v, wr, lab) in enumerate(
                [(acf_sampel(y, 12)[1:], BIRU, "ACF"),
                 (pacf_sampel(y, 12), HIJAU, "PACF")]):
            a = ax[k, j]
            a.axhspan(-b, b, color=BIRU_MUDA, lw=0)
            _stem(a, np.r_[np.nan, v], warna=wr)
            a.set_ylim(-0.5, 0.7)
            a.set_title(f"{nama}: {lab}")
            a.set_xticks([1, 4, 8, 12])
    for a in ax[1]:
        a.set_xlabel("lag $h$")
    for a in ax.flat:
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab05-identifikasi")


# ============================ Bab 6 ==================================

def bab06_ma1():
    th = np.linspace(-3, 3, 600)
    fig, ax = plt.subplots(figsize=(4.4, 2.1))
    ax.plot(th, th / (1 + th**2), color=BIRU, lw=1.0)
    ax.axvspan(-1, 1, color=BIRU_MUDA, lw=0)
    ax.axhline(1 / 12, color=MERAH, lw=0.7, ls="--")
    for v in (6 - np.sqrt(35), 6 + np.sqrt(35)):
        if v < 3:
            ax.scatter([v], [1 / 12], color=MERAH, s=12, zorder=3)
    for a, b in ((0.5, 2.0),):
        ax.scatter([a, b], [0.4, 0.4], color=JINGGA, s=12, zorder=3)
        ax.annotate(r"$\theta = 0{,}5$ dan $\theta = 2$", (2.0, 0.4),
                    (1.5, 0.52), fontsize=6, color=JINGGA)
    ax.text(0, -0.42, "dapat dibalik", ha="center", fontsize=6,
            color=BIRU)
    ax.text(2.2, 0.12, r"$r(1) = 1/12$", fontsize=6, color=MERAH)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\rho(1) = \theta/(1 + \theta^2)$")
    ax.set_ylim(-0.55, 0.62)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab06-ma1")


def bab06_arma():
    from statsmodels.tsa.arima_process import ArmaProcess
    proses = [("MA(2), $\\theta = (1, \\frac{1}{2})$", [1], [1, 1, 0.5]),
              ("ARMA(1,1), $\\phi = \\theta = \\frac{1}{2}$", [1, -0.5],
               [1, 0.5])]
    fig, ax = plt.subplots(2, 2, figsize=(4.7, 2.7))
    for k, (nama, ar, ma) in enumerate(proses):
        p = ArmaProcess(ar, ma)
        _stem(ax[k, 0], np.r_[np.nan, p.acf(9)[1:]])
        _stem(ax[k, 1], np.r_[np.nan, p.pacf(9)[1:]], warna=HIJAU)
        ax[k, 0].set_title(nama + ": ACF")
        ax[k, 1].set_title("PACF")
        for a in ax[k]:
            a.set_ylim(-0.6, 0.85)
            a.set_xticks([1, 4, 8])
    for a in ax[1]:
        a.set_xlabel("lag $h$")
    for a in ax.flat:
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab06-arma")


def bab06_balik():
    rng = np.random.default_rng(BENIH)
    T = 30
    eps = rng.standard_normal(61)
    fig, ax = plt.subplots(figsize=(4.4, 2.0))
    for th, wr in ((0.5, HIJAU), (2.0, MERAH)):
        y = eps[1:] + th * eps[:-1]
        e, seb = np.zeros(60), 0.0
        for t in range(60):
            e[t] = y[t] - th * seb
            seb = e[t]
        g = np.abs(e - eps[1:])[:T]
        ax.semilogy(np.arange(1, T + 1), g, color=wr, lw=1.0,
                    marker="o", ms=2,
                    label=f"$\\theta = {th}$".replace(".", "{,}"))
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$|e_t - \varepsilon_t|$")
    ax.legend(fontsize=6)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab06-balik")


# ============================ Bab 7 ==================================

def bab07_css():
    from bab01_data import deret_mini
    from bab07_estimasi import galat_ma1, gauss_newton_ma1
    d = deret_mini() - 4
    th = np.linspace(-0.95, 0.95, 400)
    S = [galat_ma1(d, v)[0] @ galat_ma1(d, v)[0] for v in th]
    jalur = gauss_newton_ma1(d, 0.0, 30)
    fig, ax = plt.subplots(figsize=(4.4, 2.1))
    ax.plot(th, S, color=BIRU, lw=1.0)
    for k in (0, 1, 2, 3, 5):
        ax.scatter([jalur[k][0]], [jalur[k][1]], color=MERAH, s=12,
                   zorder=3)
        ax.annotate(str(k), (jalur[k][0], jalur[k][1]),
                    (jalur[k][0] - 0.02, jalur[k][1] + 0.35), fontsize=6,
                    color=MERAH)
    ax.scatter([jalur[-1][0]], [jalur[-1][1]], color=HIJAU, s=16,
               marker="s", zorder=3)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$S(\theta) = \sum e_t^2$")
    ax.set_ylim(11, 17)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab07-css")


def bab07_loglik():
    from bab01_data import deret_mini
    y = deret_mini()
    d = y - 4
    T = len(d)
    ph = np.linspace(-0.9, 0.9, 400)
    eks, bsy = [], []
    for p in ph:
        q = (1 - p**2) * d[0] ** 2 + np.sum((d[1:] - p * d[:-1]) ** 2)
        eks.append(-T / 2 * np.log(2 * np.pi * q / T) - T / 2
                   + 0.5 * np.log(1 - p**2))
        qc = np.sum((d[1:] - p * d[:-1]) ** 2)
        bsy.append(-(T - 1) / 2 * np.log(2 * np.pi * qc / (T - 1))
                   - (T - 1) / 2)
    eks, bsy = np.array(eks), np.array(bsy)
    fig, ax = plt.subplots(figsize=(4.4, 2.1))
    ax.plot(ph, eks, color=BIRU, lw=1.0, label="eksak")
    ax.plot(ph, bsy, color=JINGGA, lw=1.0, ls="--", label="bersyarat")
    for v, wr in ((eks, BIRU), (bsy, JINGGA)):
        k = np.argmax(v)
        ax.scatter([ph[k]], [v[k]], color=wr, s=12, zorder=3)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"log-kemungkinan (profil $\sigma^2$)")
    ax.legend(fontsize=6)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab07-loglik")


def bab07_simulasi():
    import warnings
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.arima_process import ArmaProcess
    from bab07_estimasi import gauss_newton_ma1
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(BENIH)
    proses = ArmaProcess([1], [1, 0.5])
    css, mle = [], []
    for _ in range(500):
        y = proses.generate_sample(50, distrvs=rng.standard_normal)
        css.append(gauss_newton_ma1(y - y.mean(), 0.0, 30)[-1][0])
        mle.append(ARIMA(y, order=(0, 0, 1)).fit().params[1])
    fig, ax = plt.subplots(figsize=(4.4, 2.0))
    b = np.linspace(-0.2, 1.0, 37)
    ax.hist(css, bins=b, color=JINGGA, alpha=0.5, label="CSS")
    ax.hist(mle, bins=b, color=BIRU, alpha=0.5, label="MLE")
    ax.axvline(0.5, color=MERAH, lw=0.8, ls="--")
    ax.set_xlabel(r"$\hat\theta$")
    ax.legend(fontsize=6)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab07-simulasi")


# ============================ Bab 8 ==================================

def bab08_dua():
    rng = np.random.default_rng(BENIH)
    T, n = 100, 30
    t = np.arange(1, T + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.1), sharey=True)
    for _ in range(n):
        u = np.zeros(T)
        e = rng.standard_normal(T)
        for i in range(1, T):
            u[i] = 0.5 * u[i - 1] + e[i]
        a.plot(t, 0.2 * t + u, color=BIRU, lw=0.3, alpha=0.6)
        b.plot(t, np.cumsum(0.2 + rng.standard_normal(T)), color=MERAH,
               lw=0.3, alpha=0.6)
    for ax in (a, b):
        ax.plot(t, 0.2 * t, color="black", lw=0.8, ls="--")
        ax.set_xlabel("$t$")
        _rapikan(ax)
    a.set_title("tren deterministik + AR(1)")
    b.set_title("random walk + drift")
    a.set_ylabel("$y_t$")
    fig.tight_layout()
    simpan(fig, "bab08-dua")


def bab08_tau():
    import warnings
    from scipy import stats
    from statsmodels.tsa.stattools import adfuller
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(BENIH + 1)
    tau = [adfuller(np.cumsum(rng.standard_normal(100)), maxlag=0,
                    regression="c", autolag=None)[0] for _ in range(3000)]
    fig, ax = plt.subplots(figsize=(4.4, 2.0))
    ax.hist(tau, bins=50, density=True, color=BIRU_MUDA, edgecolor=BIRU,
            lw=0.4, label=r"$\tau$ di bawah $H_0$")
    g = np.linspace(-5, 3, 300)
    ax.plot(g, stats.norm.pdf(g), color=ABU, lw=0.8, label=r"$\mathcal{N}(0,1)$")
    ax.axvline(-1.645, color=ABU, lw=0.7, ls=":")
    ax.axvline(-2.86, color=MERAH, lw=0.8, ls="--")
    ax.text(-2.95, 0.47, "$-2{,}86$", fontsize=6, color=MERAH, ha="right")
    ax.text(-1.55, 0.47, "$-1{,}645$", fontsize=6, color=ABU)
    ax.set_xlabel(r"$\tau$")
    ax.legend(fontsize=5.5, loc="upper right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab08-tau")


def bab08_palsu():
    rng = np.random.default_rng(BENIH + 2)
    T = 100
    a = np.cumsum(rng.standard_normal(T))
    b = np.cumsum(rng.standard_normal(T))
    fig, (k, l) = plt.subplots(1, 2, figsize=(4.7, 2.0))
    k.plot(a, color=BIRU, lw=0.7, label="$x_t$")
    k.plot(b, color=JINGGA, lw=0.7, label="$y_t$")
    k.set_xlabel("$t$")
    k.legend(fontsize=6)
    k.set_title("dua random walk bebas")
    l.scatter(a, b, s=4, color=ABU)
    w = np.polyfit(a, b, 1)
    g = np.linspace(a.min(), a.max(), 10)
    l.plot(g, np.polyval(w, g), color=MERAH, lw=0.9)
    l.set_xlabel("$x_t$")
    l.set_ylabel("$y_t$")
    l.set_title("garis regresi palsu")
    for ax in (k, l):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab08-palsu")


# ============================ Bab 9 ==================================

def bab09_kipas():
    from bab01_data import deret_mini, ramal
    from bab09_ramalan import galat_baku_ramalan
    y = deret_mini()
    H = 10
    f = ramal(y, np.array([4, 0.5, -0.5]), H)
    se = galat_baku_ramalan([0.5, -0.5], [], 1.0, H)
    tt = np.arange(9, 9 + H)
    fig, ax = plt.subplots(figsize=(4.7, 2.1))
    ax.fill_between(np.r_[8, tt], np.r_[3, f - 1.96 * se],
                    np.r_[3, f + 1.96 * se], color=HIJAU_MUDA, lw=0,
                    label="selang 95%")
    ax.fill_between(np.r_[8, tt], np.r_[3, f - 1.2816 * se],
                    np.r_[3, f + 1.2816 * se], color="#BFDCD3", lw=0,
                    label="selang 80%")
    ax.plot(np.arange(1, 9), y, color=BIRU, lw=1.0, marker="o", ms=3)
    ax.plot(np.r_[8, tt], np.r_[3, f], color=HIJAU, lw=1.0, marker="s",
            ms=2.5, label="ramalan")
    ax.axhline(4, color=ABU_GARIS, lw=0.6, ls="--")
    ax.set_xticks(np.arange(1, 19))
    ax.set_xlabel("pekan $t$")
    ax.set_ylabel("$y_t$")
    ax.legend(fontsize=5.5, loc="upper left", ncol=3)
    ax.set_ylim(0, 8.5)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab09-kipas")


def bab09_lebar():
    from bab09_ramalan import galat_baku_ramalan
    H = 20
    h = np.arange(1, H + 1)
    fig, ax = plt.subplots(figsize=(4.4, 2.0))
    ax.plot(h, galat_baku_ramalan([0.5, -0.5], [], 1.0, H), color=BIRU,
            marker="o", ms=2.5, lw=0.9, label="AR(2) data mini")
    ax.plot(h, galat_baku_ramalan([1.0], [], 1.0, H), color=MERAH,
            marker="s", ms=2.5, lw=0.9, label=r"random walk, $\sqrt{h}$")
    ax.axhline(np.sqrt(1.5), color=BIRU, lw=0.6, ls="--")
    ax.text(20, np.sqrt(1.5) - 0.35, r"$\sqrt{\gamma(0)} = 1{,}22$",
            fontsize=6, color=BIRU, ha="right")
    ax.set_xlabel("horizon $h$")
    ax.set_ylabel("galat baku ramalan")
    ax.legend(fontsize=6)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab09-lebar")


def bab09_cakupan():
    import subprocess
    keluaran = subprocess.run([sys.executable, "kode/bab09_simulasi.py"],
                              capture_output=True, text=True).stdout
    baris = [b for b in keluaran.splitlines() if "parameter" in b]
    nilai = [[float(v) for v in b.split(":")[1].split()] for b in baris]
    H = [1, 2, 5, 10]
    x = np.arange(len(H))
    fig, ax = plt.subplots(figsize=(4.4, 2.0))
    lebar = 0.2
    warna = [BIRU, JINGGA, HIJAU, MERAH]
    nama = ["$T = 30$, benar", "$T = 30$, taksir", "$T = 200$, benar",
            "$T = 200$, taksir"]
    for k in range(4):
        ax.bar(x + (k - 1.5) * lebar, nilai[k], width=lebar,
               color=warna[k], label=nama[k])
    ax.axhline(0.95, color=ABU, lw=0.7, ls="--")
    ax.set_ylim(0.88, 0.97)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$h = {h}$" for h in H])
    kunci_label(ax, "x")
    ax.set_ylabel("cakupan")
    ax.legend(fontsize=5, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab09-cakupan")


# Fungsi gambar baru disisipkan DI ATAS penanda ini.


if __name__ == "__main__":
    pola = re.compile(r"^bab\d\d_")
    pilihan = sys.argv[1:]
    fungsi = [(k, v) for k, v in sorted(globals().items())
              if pola.match(k) and callable(v)]
    if pilihan:
        fungsi = [(k, v) for k, v in fungsi
                  if any(p in k for p in pilihan)]
    if not fungsi:
        print("tidak ada gambar yang cocok")
    for _, f in fungsi:
        f()
