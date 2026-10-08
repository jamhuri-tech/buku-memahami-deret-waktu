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
