# Kode — Memahami Deret Waktu

Repositori pendamping buku **_Memahami Deret Waktu: ARIMA dan
Peramalan, Periode demi Periode_** (Edisi Pertama, 2026) oleh Mohammad
Jamhuri, seri *Memahami*. Naskah buku sedang ditulis; repositori ini
diperbarui setiap kali satu bab selesai.

Buku ini memakai satu **data mini** dari awal sampai akhir: penjualan
mingguan delapan pekan, y = 2, 5, 6, 5, 3, 4, 4, 3, dengan rata-rata 4.
Disusun sebagai tabel lag (x1 = y(t-1), x2 = y(t-2), target y(t)),
kuadrat terkecil AR(2)-nya tepat w = (4, 1/2, -1/2). Setiap
autokorelasi, taksiran, uji, ramalan, dan selang ramalan dihitung
tangan di buku lalu diperiksa oleh kode di sini. File
`kode/babNN_contoh.py` memeriksa setiap bilangan di kotak Contoh Soal
Bab NN.

**Setiap angka keluaran yang tercetak di buku dihasilkan oleh kode di
sini**, dan `periksa.py` membuktikannya: skrip itu menjalankan ulang
kode setiap bab dan mencocokkan hasilnya dengan blok keluaran yang
tercetak di buku.

Seluruh kode boleh dipakai, disalin, diubah, dan disebarluaskan secara
bebas untuk keperluan apa pun, termasuk komersial, tanpa kewajiban
mencantumkan sumber (lisensi [0BSD](LICENSE)). Data nyata mempunyai
lisensinya sendiri, tercatat di `data/SUMBER.md`.

## Menjalankan

```bash
git clone https://github.com/jamhuri-tech/buku-memahami-deret-waktu.git
cd buku-memahami-deret-waktu
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python kode/bab01_deret.py
```

Setiap skrip dijalankan dari akar repositori. Pembangkit bilangan acak
selalu memakai benih tetap (20261008), sehingga keluarannya sama setiap
kali dijalankan.

## Memeriksa angka di buku

```bash
.venv/bin/python periksa.py        # semua bab
.venv/bin/python periksa.py 01     # Bab 1 saja
```

Keluaran `SEMUA COCOK` berarti setiap blok keluaran di buku dihasilkan
ulang oleh kode ini. Pemeriksaan yang sama berjalan otomatis di GitHub
Actions setiap kali kode berubah.

## Struktur

```
kode/
  bab01_data.py      data mini, tabel lag, kuadrat terkecil, ramalan
  babNN_*.py         kode Bab NN
  babNN_contoh.py    pemeriksa hitungan tangan Contoh Soal Bab NN
data/SUMBER.md       asal setiap data: alamat, tanggal, lisensi
keluaran/babNN.txt   blok keluaran yang tercetak di Bab NN
gen_gambar.py        membangkitkan semua gambar buku ke gbr/
periksa.py           mencocokkan kode dengan keluaran/
requirements.txt     versi pustaka yang dipakai buku
```

`python gen_gambar.py bab01` membangkitkan gambar Bab 1 saja.

## Versi

Tag `edisi-1` akan menandai kode yang tepat dipakai untuk mencetak
Edisi Pertama. `requirements.txt` mematok versi yang dipakai buku
(Python 3.13, NumPy 2.1.3, SciPy 1.15.3, statsmodels 0.14.4, pandas
2.2.3, Matplotlib 3.10.0). Versi lain biasanya berjalan, tetapi digit
terakhir sebagian angka dapat berbeda.

## Salah ketik, galat, dan saran

Silakan buka [issue](../../issues) di repositori ini: sebutkan bab,
halaman, dan apa yang keliru. Saran juga dapat dikirim ke
m.jamhuri@live.com.
