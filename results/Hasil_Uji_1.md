# Hasil Uji 1 — horizon dan kontrol sakit tetapi hidup

Dijalankan 26 September 2026. Skrip `01_denyut.py`, `02_fitur.py`, `03_uji.py`; log lengkap di
`keluaran/uji.log` (tidak masuk repo), tabel di `keluaran/hasil_uji.csv`.

**Rancangan singkat.** Ciri HRV per jendela 5 menit (11 ciri, sama dengan pustaka). Pembagian
**tingkat subjek**, 5 lipatan diulang 20 kali. AUC atas subjek, selang 95 % bootstrap subjek.
SCD = 20 rekaman `sddb` yang punya VF. Kontrol sehat = 18 `nsrdb` (128 Hz). Kontrol sakit tetapi
hidup = 15 `chfdb` (gagal jantung berat, **250 Hz seperti `sddb`**).

## Temuan

### 1. AUC SCD lawan orang sehat tidak turun sampai 12 jam sebelum VF

| Menit sebelum VF | 0 | 30 | 60 | 120 | 180 | 360 | 720 |
|---|---|---|---|---|---|---|---|
| n SCD | 20 | 20 | 20 | 19 | 16 | 12 | 8 |
| Logistik, semua ciri | 0,95 | 0,92 | 0,95 | 0,90 | 0,93 | 0,89 | 0,91 |
| Random forest, semua ciri | 1,00 | 0,97 | 0,94 | 0,93 | 0,96 | 0,90 | 0,99 |

Pengklasifikasi yang benar-benar melihat kematian yang mendekat seharusnya melemah menjauhi
awitan. Kurvanya datar. Rancangan tolok ukur ini — rancangan hampir seluruh pustaka yang
diaudit — **tidak mampu membedakan "akan meninggal dalam lima menit" dari "akan meninggal
dalam dua belas jam"**, dan karena itu tidak mengukur apa yang diklaimnya.

⚠️ n pada horizon 6 dan 12 jam kecil (12 dan 8) dan selangnya lebar ([0,70, 1,00]). Yang kuat
adalah **tiadanya penurunan**, bukan nilai titiknya.

### 2. Model tolok ukur menyatakan pasien gagal jantung yang hidup "akan meninggal"

Model yang dilatih SCD (jendela 0-30 menit sebelum VF) lawan `nsrdb`, lalu dipakai pada 15
pasien `chfdb`:

| Model | Ciri | Pasien CHF dinyatakan SCD |
|---|---|---|
| Logistik | semua | **80 %** |
| Logistik | tanpa ektopik | 73 % |
| Random forest | semua | **93 %** |
| Random forest | tanpa ektopik | 87 % |

Pasien-pasien ini tidak meninggal selama rekaman. Model yang dilatih pada rancangan pustaka
mendeteksi "pasien jantung sakit", bukan "kematian yang akan datang".

### 3. Satu ciri remeh sudah cukup

Proporsi denyut ektopik saja: AUC SCD lawan sehat **0,94** (logistik) dan 0,95 (RF) pada 0 menit,
dan **0,99-1,00 pada 12 jam**. Lawan CHF, ciri yang sama jatuh ke 0,65 lalu di bawah 0,5 menjauhi
VF — pasien CHF punya denyut ektopik sebanyak pasien SCD. Artinya: pada rancangan SCD lawan
sehat, "pasien punya ektopik" sudah menjawab hampir seluruh soal.

### 4. Dengan kontrol sakit pada frekuensi cuplik sama, pemisahannya lebih kecil dan melemah

SCD lawan CHF, semua ciri: logistik 0,89 → 0,90 → 0,83 → 0,78 → 0,82 → 0,71 dari 0 menit ke 12
jam; RF 0,88 → 0,89 → 0,92 → 0,87 → 0,80 → 0,85 → 0,73.

Ini estimasi yang lebih jujur, dan ia **tidak nol** — ada pemisahan sekitar 0,8-0,9 dalam dua
jam pertama, dengan kecenderungan turun menjauhi VF. ⚠️ Tetapi `sddb` dan `chfdb` juga dua
basis data berbeda (rumah sakit, era, alat), jadi rancu belum hilang seluruhnya, hanya frekuensi
cuplik yang disamakan. Selang lebar; kecenderungan turunnya belum teruji secara formal.

## Catatan teknis yang harus ditulis di naskah

- AUC di bawah 0,5 untuk ciri laju jantung saja (misalnya 0,06 pada 120 menit) adalah artefak
  validasi silang yang dikenal pada ciri lemah dengan sampel kecil (*anti-learning*), bukan
  sinyal terbalik. Laporkan sebagai "tanpa informasi".
- Anotasi `sddb` yang dipakai `.ari` (belum diaudit) untuk semua rekaman; `.atr` teraudit hanya 9.
- Waktu dalam sehari tidak disamakan: jendela `nsrdb` dan `chfdb` diambil acak, jendela `sddb`
  mengikuti jam awitan. Uji lanjut: samakan jam.

## Langkah berikut

1. Uji awitan palsu: pindahkan awitan VF ke titik acak, ulangi uji 1.
2. Uji kecenderungan formal atas horizon (regresi AUC lawan horizon dengan bootstrap).
3. Samakan jam dalam sehari.
4. Replikasi dua rancangan bocor dari pustaka (Lai 2019, Reddy 2022): pembagian segmen lawan
   subjek, pada data dan ciri yang sama.
5. Gambar: kurva AUC lawan horizon, tiga garis (lawan sehat, lawan CHF, ektopik saja).

---

# Hasil Uji 2 — kecenderungan, awitan palsu, rancangan bocor

Dijalankan 26 September 2026. Skrip `04_uji_lanjut.py` dan `05_bocor.py`; log di
`keluaran/uji_lanjut.log`, tabel di `keluaran/hasil_uji_lanjut.csv` dan `keluaran/hasil_bocor.csv`.

## 5. Kecenderungan horizon pada kohort tetap (12 SCD yang punya data 0-6 jam)

| Model, kontrol | AUC 0 / 0,5 / 1 / 2 / 3 / 6 jam | Kemiringan per jam [95 %] | AUC(0) − AUC(6 jam) [95 %] |
|---|---|---|---|
| Logistik, nsrdb | 0,90 / 0,88 / 0,90 / 0,85 / 0,88 / 0,89 | −0,000 [−0,007, +0,004] | +0,009 [−0,014, +0,056] |
| RF, nsrdb | 1,00 / 0,93 / 0,89 / 0,89 / 0,93 / 0,89 | −0,010 [−0,027, +0,001] | +0,106 [+0,000, +0,273] |
| Logistik, chfdb | 0,86 / 0,87 / 0,90 / 0,80 / 0,76 / 0,81 | −0,014 [−0,035, +0,002] | +0,044 [−0,067, +0,161] |
| RF, chfdb | 0,87 / 0,84 / 0,92 / 0,84 / 0,76 / 0,86 | −0,007 [−0,029, +0,013] | +0,011 [−0,078, +0,122] |

Dengan subjek yang sama di semua horizon, kurva logistik lawan orang sehat **datar dan
selangnya sempit**: penurunan enam jam paling banyak 0,056. Random forest memperlihatkan satu
pengecualian — AUC 1,00 tepat sebelum VF lalu 0,89-0,93 sesudahnya, selisih +0,11 dengan batas
bawah tepat nol. Itu konsisten dengan irama ventrikel yang memang muncul di menit-menit terakhir
sebelum VF, bukan dengan tanda peringatan berjam-jam.

## 6. Awitan palsu hampir sama baiknya dengan awitan sebenarnya

Awitan dipindah acak ke paling lambat 60 menit sebelum VF sebenarnya, 20 undian:

| Model | Kontrol | AUC awitan sebenarnya (uji 1) | AUC awitan palsu, rerata [min-maks] |
|---|---|---|---|
| Logistik | nsrdb | 0,953 | 0,916 [0,858-0,947] |
| RF | nsrdb | 0,997 | 0,937 [0,907-0,985] |
| Logistik | chfdb | 0,890 | 0,822 [0,741-0,874] |
| RF | chfdb | 0,883 | 0,842 [0,747-0,937] |

Label waktu menyumbang sekitar 0,04-0,07 AUC. Sisanya — sekitar 0,92 lawan orang sehat dan 0,83
lawan pasien gagal jantung — tersedia dari **jam mana pun** dalam rekaman pasien yang kelak
meninggal. Itu bukan ramalan; itu pengenalan pasien.

## 7. Rancangan bocor pustaka: lima sampai sembilan poin

Rancangan Lai 2019 dan Reddy 2022 (30 segmen satu menit per rekaman, 20 SCD dan 18 normal):

| Model | Bocor (lipatan atas segmen) | Benar (lipatan atas rekaman) | Selisih |
|---|---|---|---|
| Random forest | 93,9 % | 88,5 % | −5,4 poin |
| KNN | 92,7 % | 83,7 % | −9,0 poin |
| SVM | 92,3 % | 87,3 % | −5,0 poin |

Dua catatan jujur. Pertama, angka bocor di sini 92-94 %, bukan 99-100 % seperti di kedua naskah;
cirinya tidak identik (keduanya memakai EMD/DWT, di sini HRV baku), jadi yang direplikasi
rancangan pembagiannya, bukan angkanya. Kedua, dengan pembagian benar pun akurasinya masih 84-89
% — **kebocoran menjelaskan sebagian kecil; rancu basis data menjelaskan sisanya**, dan itu
tidak bisa diperbaiki dengan pembagian yang benar.

## Ringkasan untuk naskah

1. Rancangan tolok ukur SCD lawan orang sehat memisahkan kelas sama baiknya enam dan dua belas jam
   sebelum VF seperti lima menit sebelumnya.
2. Memindah awitan ke waktu acak hanya menurunkan AUC 0,04-0,07.
3. Model tolok ukur menyatakan 80-93 % pasien gagal jantung yang hidup sebagai "akan meninggal".
4. Satu ciri remeh — proporsi denyut ektopik — mencapai AUC 0,94.
5. Kebocoran pembagian menambah 5-9 poin; sisanya rancu basis data yang bertahan di bawah
   pembagian yang benar.
6. Dengan kontrol sakit pada frekuensi cuplik sama, pemisahan sekitar 0,8-0,9 tetap ada, dengan
   sedikit kecenderungan turun (−0,007 sampai −0,014 per jam, selang melewati nol). Itu batas atas
   yang jujur untuk sinyal yang mungkin nyata, dan masih bercampur dengan perbedaan basis data.

---

## Tambahan 27 September 2026 — uji dalam-pasien sddb (`07_dalam_pasien.py`)

Menjawab keberatan efek langit-langit. Jendela 0-30 menit sebelum VF lawan jendela pasien yang
sama >= 2, 3, atau 6 jam sebelum VF (19, 16, 13 pasien):

| Kontrol | Logistik | RF | RF tanpa ektopik |
|---|---|---|---|
| >= 2 jam | 0,533 [0,42, 0,64] | 0,603 [0,51, 0,70] | 0,548 |
| >= 3 jam | 0,500 [0,40, 0,62] | 0,608 [0,52, 0,72] | 0,578 |
| >= 6 jam | 0,575 [0,46, 0,71] | 0,591 [0,43, 0,74] | 0,519 |

Di dalam pasien yang sama, sisa sinyalnya kecil dan dibawa ektopi setengah jam terakhir, jauh
dari 0,9-1,0 yang dilaporkan tolok ukur.

## Tambahan 28 September 2026 — anotasi teraudit (`08_anotasi_teraudit.py`)

Menjawab isu telaah panel (tiga peran): fraksi ektopik sddb dihitung dari `.ari` otomatis, nsrdb
dari `.atr` teraudit. **Koreksi fakta:** `.atr` teraudit penuh ada untuk 12 rekaman sddb (30, 31,
32, 34, 35, 36, 41, 45, 46, 49, 51, 52), 11 di antaranya dengan awitan VF — bukan 9 seperti
tertulis di draf. Analisis diulang pada 11 pasien itu, sekali dengan `.ari` dan sekali dengan
`.atr`; kontrol nsrdb/chfdb identik (benih sama). Log lengkap: `keluaran/teraudit.log`.

Kesepakatan per jendela (1.447 jendela bersama): Spearman 0,85-1,00 untuk kesebelas ciri,
fraksi ektopik 0,86. `.atr` membuang lebih banyak jendela (1.449 lawan 1.633), jadi pasien per
horizon lebih sedikit (9 lawan 11 pada awitan).

| Analisis | Jam | `.ari` | `.atr` |
|---|---|---|---|
| Logistik, semua ciri, lawan nsrdb | 0 | 0,909 [0,70, 1,00] n=11 | 0,889 [0,66, 1,00] n=9 |
| Logistik, semua ciri, lawan nsrdb | 6 | 0,826 [0,55, 1,00] n=8 | 0,786 [0,50, 0,99] n=7 |
| RF, semua ciri, lawan nsrdb | 0 | 1,000 [1,00, 1,00] | 0,988 [0,95, 1,00] |
| RF, semua ciri, lawan nsrdb | 6 | 0,854 [0,56, 1,00] | 0,841 [0,50, 1,00] |
| Logistik, ektopik saja, lawan nsrdb | 0 | 0,899 [0,69, 1,00] | 0,889 [0,67, 1,00] |
| Logistik, ektopik saja, lawan nsrdb | 6 | 0,701 [0,36, 1,00] | 0,714 [0,33, 1,00] |
| RF dalam-pasien, >= 2 jam | — | 0,579 [0,49, 0,71] n=11 | 0,584 [0,51, 0,69] n=9 |

Kesimpulan: pemisahan dan sinyal fraksi ektopik tidak bergantung pada anotasi otomatis. Lawan
chfdb, fraksi ektopik saja di bawah 0,5 pada 1-6 jam sebelum VF dengan kedua sumber untuk
logistik (`.ari` 0,38-0,46; `.atr` 0,02-0,22) dan untuk RF kecuali `.atr` 3 jam (0,51) — sddb
lebih sedikit ektopik daripada gagal jantung jam-jam sebelum VF, sama dengan analisis utama. Yang belum:
deteksi ulang denyut dari sinyal mentah dengan satu detektor untuk semua basis data.

---

## Analisis telaah ulang panel 9 Oktober 2026 (`09_revisi_panel.py`, `keluaran/revisi_panel.log`)

Pemicu: `../Telaah/2026-10-08-EKG1/report.md` (major revision).

**Tanda irama per jendela** (`keluaran/irama_jendela.csv`, dari aux_note dan runtun V ≥ 3):
sddb 2.508 jendela pra-VF — AF/flutter 36,7 % (14 dari 20 rekaman), VT 3,7 % (11 rekaman), pacu 0;
nsrdb 0 / 0 / 0; chfdb AF 6,4 % (4 rekaman), VT 2,6 % (11 rekaman).

**Uji surogat awitan palsu yang benar** (19 pasien sama; p peringkat = (1 + #palsu ≥ sebenarnya)/(N + 1)):
logistik lawan nsrdb 0,965 lawan palsu 0,931 (+0,033), p 0,080 (N = 99); logistik lawan chfdb 0,846
lawan 0,772 (+0,073), p 0,010; RF lawan nsrdb 1,000 lawan 0,955 (+0,045), p 0,025 (N = 39); RF lawan
chfdb 0,832 lawan 0,812 (+0,019), p 0,375. Awitan sebenarnya membawa sinyal kecil (+0,02 sampai +0,07)
pada dua dari empat konfigurasi.

**Pita 10 menit.** Lawan nsrdb: 0,95–1,00 di setiap pita 0–10, 10–20, 20–30, 30–60 menit, 0,92–0,96
pada 120–150 dan 360–390 menit. Lawan chfdb: 0,78–0,89 tanpa arah. Dalam-pasien per pita 10 menit
lawan ≥ 120 menit: logistik 0,525–0,589, RF 0,573–0,607; semua IK mencakup 0,5 (19 pasien).

**Tanpa VT:** tidak berubah (lawan nsrdb 0,947 / 1,000 pada h = 0; dalam-pasien 0,577 / 0,546).
**Sinus saja** (tanpa AF/flutter, pacu, VT; sddb tinggal 1.532 jendela): lawan nsrdb 0,924 / 1,000
(h = 0, 11 pasien), 0,889 / 0,872 (h = 360, 10 pasien); dalam-pasien 0,430 / 0,594 (10 pasien). AF
tidak menjelaskan pemisahan antarbasis data.

**Dalam-pasien berpasangan** (per pasien rerata skor OOF dekat − jauh, 19 pasien): logistik dekat > jauh
11/19, median +0,003, Wilcoxon p 0,441; RF 13/19, median +0,096, Wilcoxon p 0,036 (tanpa ektopik
+0,070, p 0,014); uji tanda p 0,167.

**Bootstrap dengan latih ulang, kohort tetap 12 pasien, AUC(0) − AUC(360):** logistik lawan nsrdb
+0,000 [−0,191; +0,190] (B = 200); lawan chfdb +0,011 [−0,178; +0,283]; RF lawan nsrdb +0,116
[+0,002; +0,381] (B = 60); lawan chfdb +0,050 [−0,148; +0,272]. Selang jauh lebih lebar daripada
bootstrap tanpa latih ulang; batas "paling banyak 0,06" di naskah tidak bertahan.

**Simulasi daya** (perubahan pra-VF buatan δ SD ke arah sddb − nsrdb, meluruh e^(−t/60 menit), bobot
per pasien N(1; 0,5), 20 simulasi, logistik):

| δ (SD) | AUC(0) − AUC(360) rerata | uji horizon menangkap | uji dalam-pasien menangkap |
|---|---|---|---|
| 0 | +0,013 | 0/20 | 3/20 |
| 0,5 | +0,045 | 0/20 | 15/20 |
| 1,0 | +0,071 | 0/20 | 20/20 |
| 2,0 | +0,081 | 0/20 | 20/20 |

Bacaan:
- **Uji horizon lawan nsrdb tidak punya daya sama sekali** pada langit-langit AUC ~0,95: perubahan
  pra-VF buatan 2 SD tidak pernah terdeteksi. Kurva horizon yang datar karena itu bukan bukti bahwa
  tidak ada perubahan pra-VF — dan juga berarti kurva datar terbitan tidak bisa menjadi bukti waktu
  peringatan. Klaim naskah harus berpindah: desain sddb-lawan-nsrdb secara struktural tidak mampu
  mengukur waktu peringatan.
- **Uji dalam-pasien punya daya** untuk perubahan ≥ 0,5–1 SD (15/20, 20/20), dengan positif palsu 3/20
  pada δ = 0 (bootstrap tanpa latih ulang terlalu sempit). Hasil nyatanya kecil: logistik nol, RF
  ~0,6 dan berpasangan +0,10 (p 0,036). Jadi: paling banyak perubahan kecil dalam 30 menit terakhir.
- Surogat: awitan sebenarnya sedikit lebih baik daripada awitan palsu pada dua konfigurasi. Konsisten
  dengan sinyal kecil di dekat VF; tidak mendukung peringatan berjam-jam.

**Diagnosis simulasi daya (9 Oktober 2026, sesudah telaah panel ketiga `../Telaah/2026-10-09-EKG1/`).**
Tiga simulasi pertama per δ diulang dengan selang dicetak (skrip sementara, logika sama dengan
`daya()`): δ = 0 → selisih +0,009 / +0,009 / +0,014, selang [−0,00; 0,046], [−0,012; 0,046],
[−0,009; 0,065]; δ = 2 → +0,069 / +0,093 / +0,088, selang [0,000; 0,208], [0,000; 0,259], [0,000; 0,250];
AUC(0) naik ke 0,98–1,00. **Bukan bug pada pengambilan subjek** (30 subjek, 12 positif, tanpa NaN),
tetapi juga **bukan "tanpa daya"**: titik estimasi bergeser jelas, dan batas bawah selang menempel tepat
di 0 karena pada langit-langit banyak resampel memberi AUC 1 di kedua horizon (selisih 0); aturan
deteksi "persentil 2,5 > 0" (ketat) lalu tidak pernah terpenuhi. Klaim naskah "no power at all" /
"never detected" terlalu kuat dan bergantung pada aturan deteksi; perlu simulasi ulang (lihat rencana).

**Pemeriksaan AF sddb (`10_daya_ulang.py af`, `keluaran/daya_ulang_af.log`, 9 Oktober 2026).** Anotasi
otomatis `.ari` memuat tanda irama `(AFIB` di 22 dari 23 rekaman (hanya 51 tanpa), padahal dokumentasi
PhysioNet menyebut 4 pasien AF. Lama AF menurut `.ari` (tanda irama saja, tanda ST diabaikan): ≥ 70 %
rekaman pada 35 (1.482 dari 1.492 menit), 50 (1.345/1.388) dan 36 (890/1.221); 34–36 % pada 33 dan 44;
selebihnya puluhan menit. Berkas `.atr` teraudit (12 rekaman) tidak memuat tanda irama sama sekali.
Jadi tanda AF/flutter per jendela (36,7 % jendela pra-VF, 14 rekaman) berasal dari detektor otomatis
yang tidak diaudit dan kemungkinan besar berlebih; subset "sinus saja" adalah "jendela yang tidak
ditandai AF oleh anotasi otomatis". Naskah bagian 4 dan 6.4 harus menyatakan ini.

**Uji permutasi dalam-pasien** (`10_daya_ulang.py permutasi`, `keluaran/daya_ulang_permutasi.log`; dekat
0–30 menit lawan ≥ 2 jam, 19 pasien, label diacak di dalam pasien, model dilatih ulang tiap permutasi):
logistik AUC 0,560, permutasi rerata 0,496 [0,411; 0,571], p 0,075 (N = 199); random forest AUC 0,608,
permutasi 0,498 [0,430; 0,566], p 0,010 (N = 99, nilai terkecil yang mungkin). Sinyal kecil random forest
bertahan pada uji yang tingkat galatnya benar; logistik tidak.

**Simulasi daya ulang, regresi logistik** (`10_daya_ulang.py daya`, `keluaran/daya_ulang.csv`,
`keluaran/daya_ulang_logistik.log`). Horizon: kohort tetap lawan nsrdb, AUC(0) − AUC(6 jam), 500 resampel
tanpa latih ulang, terdeteksi bila p dua sisi < 0,05 dengan seri di 0 dihitung setengah. Dalam-pasien:
Wilcoxon satu sisi tingkat pasien (dekat > jauh), p < 0,025. δ = 0 adalah data nyata tanpa suntikan
(bukan nol yang dijamin): logistik 0/100 pada kedua uji.

| Arah perubahan pra-VF | Besar | Selisih AUC rerata | Horizon terdeteksi | Dalam-pasien terdeteksi |
|---|---|---|---|---|
| tanpa suntikan | 0 | +0,015 | 0/100 | 0/100 |
| searah sddb − nsrdb | 0,5 / 1 / 2 SD | +0,045 / +0,066 / +0,077 | 0 / 0 / 0 dari 50 | 23 / 50 / 50 |
| berlawanan | 0,5 / 1 / 2 SD | −0,051 / −0,223 / −0,611 | 7 / 44 / 50 | 0 / 50 / 50 |
| tegak lurus | 0,5 / 1 / 2 SD | +0,032 / +0,061 / +0,076 | 0 / 0 / 0 | 50 / 50 / 50 |
| denyut jantung naik | +5 / +10 / +20 per menit | +0,014 / +0,019 / +0,034 | 0 / 0 / 0 | 0 / 39 / 50 |

Lebar selang horizon rerata naik dari 0,06 (tanpa suntikan) ke 0,23 (searah 2 SD) dan 0,47 (berlawanan
2 SD). Bacaan:
- Uji horizon lawan nsrdb **hanya** menangkap perubahan yang membuat sddb lebih mirip kontrol (AUC turun).
  Perubahan searah perbedaan basis data, tegak lurus terhadapnya, atau kenaikan denyut jantung sampai
  +20/menit tidak pernah tertangkap: AUC sudah dekat langit-langit dan selangnya melebar. Klaim yang benar:
  "kurva horizon datar tidak menyingkirkan perubahan pra-VF pada sebagian besar arah", bukan "tidak punya
  daya ke arah mana pun".
- Uji dalam-pasien menangkap perubahan ≥ 1 SD ke semua arah dan kenaikan denyut ≥ 10/menit (39/50) sampai
  20/menit (50/50); +5/menit (seukuran yang dilaporkan sebelum VT, Shusterman 1998) tidak tertangkap.
- Asimetri pada 0,5 SD (searah 23/50, berlawanan 0/50, tegak 50/50) belum dijelaskan.

**Artefak seri dan simulasi ulang dengan ciri kontinu saja** (`keluaran/daya_ulang_rf.log`,
`keluaran/daya_ulang_kontinu.log`). frac_ektopik bernilai tepat 0 pada 97,7 % jendela nsrdb dan 34 % sddb
(ciri ini tidak dilog), pnn50 = 0 pada 13 % sddb. Suntikan searah ciri itu memindahkan nilai seri sddb
sekecil apa pun, dan random forest lalu memisahkan sddb dari nsrdb di semua horizon: AUC(6 jam) RF naik
dari 0,904 ke 0,989 walau suntikan di 6 jam hanya ~0,25 % (artefak). Simulasi diulang dengan suntikan
hanya pada sembilan ciri kontinu (arah `_k`; frac_ektopik dan pnn50 tidak disentuh). Horizon dihitung dua
sisi (p < 0,05, seri di 0 setengah). Dalam-pasien: Wilcoxon satu sisi tingkat pasien.

Regresi logistik, 50 simulasi per sel (horizon / dalam-pasien terdeteksi):

| Arah (ciri kontinu) | 0,5 SD | 1 SD | 2 SD | Selisih AUC rerata pada 2 SD |
|---|---|---|---|---|
| searah sddb − nsrdb | 0 / 47 | 0 / 50 | 0 / 50 | +0,075 |
| berlawanan | 0 / 31 | 2 / 50 | 0 / 50 | −0,078 |
| tegak lurus | 0 / 50 | 0 / 50 | 0 / 50 | +0,074 |
| denyut jantung +5 / +10 / +20 per menit | 0 / 0 | 0 / 39 | 0 / 50 | +0,034 (pada +20) |

Random forest, 10 simulasi per sel: AUC(0) 0,994–1,000 dan AUC(6 jam) 0,904 di setiap sel tanpa artefak;
horizon terdeteksi 0–1 dari 10 di semua sel (searah_k, balik_k, tegak_k 1 dan 2 SD, denyut +20; tanpa
suntikan 1/10); dalam-pasien 9–10 dari 10 di semua sel, termasuk tanpa suntikan (sinyal nyata, sesuai
permutasi p 0,010), jadi simulasi RF tidak bisa menunjukkan daya dalam-pasien di atas sinyal nyata.

Bacaan akhir:
- Untuk perubahan pra-VF pada ciri HRV dan denyut jantung, uji horizon lawan nsrdb **tidak menangkap
  apa pun ke arah mana pun** (≤ 2/50 logistik, ≤ 1/10 RF); selisih AUC tetap < 0,08 karena pemisahan
  antarbasis data dibawa frac_ektopik, yang tidak berubah. Satu-satunya pengecualian (simulasi pertama,
  arah berlawanan dengan frac_ektopik): perubahan yang menurunkan ektopi sddb ke arah nsrdb tertangkap
  (44/50 pada 1 SD). Klaim naskah: kurva horizon datar tidak menyingkirkan perubahan pra-VF pada HRV dan
  laju jantung; ia hanya peka terhadap perubahan ektopi yang membuat sddb mirip kontrol.
- Uji dalam-pasien (Wilcoxon tingkat pasien) menangkap perubahan ≥ 1 SD ke semua arah (50/50), 31–50/50
  pada 0,5 SD, dan kenaikan denyut ≥ +10 per menit (39/50); +5 per menit tidak.
- δ = 0 logistik 0/100 pada kedua uji; asimetri 0,5 SD pada simulasi pertama (23 / 0 / 50) sebagian besar
  artefak seri (kontinu: 47 / 31 / 50).

**Daya disesuaikan ukuran (10 Oktober 2026, sesudah telaah panel keempat `../Telaah/2026-10-10-EKG1/`).**
Dari `daya_ulang.csv` (logistik): pita nol dari 100 simulasi tanpa suntikan, selisih AUC(0) − AUC(6 jam)
di persentil 2,5–97,5 = [0,000; 0,030]. Simulasi yang selisihnya keluar dari pita: searah_k 37 / 41 / 46
dari 50 (0,5 / 1 / 2 SD), balik_k 40 / 49 / 49, tegak_k 29 / 41 / 45, denyut +5 / +10 / +20: 2 / 6 / 34;
tanpa suntikan 3/100. **Catatan:** semua simulasi memakai 12 pasien yang sama (hanya undian kontrol,
benih CV, dan bobot suntikan yang berubah), jadi pita nol ini tidak memuat ketidakpastian pengambilan
sampel pasien, dan 100 simulasi tanpa suntikan bukan 100 ulangan nol yang bebas. Bacaan: suntikan
memang menggeser titik estimasi uji horizon (+0,04 sampai +0,08), tetapi geseran itu jauh di dalam
ketidakpastian akibat hanya 12 pasien (selang tanpa latih ulang 0,06–0,23, dengan latih ulang ±0,19).
Kegagalan uji horizon di simulasi adalah soal ukuran sampel dan langit-langit, bukan ketidakpekaan
mutlak; mekanisme "ektopi membawa pemisahan" belum diuji.

**Demografi dari dokumentasi PhysioNet dan header** (10 Oktober 2026). sddb (tabel klinis di
physionet.org/content/sddb/1.0.0/, diambil 10 Okt 2026; "Patient information is limited, and sometimes
completely unavailable"): umur tercatat 19 dari 23, median 68 (17–89); 13 laki-laki, 8 perempuan, 2 tak
diketahui; riwayat tercatat pada 12 (gagal jantung 2, bedah pintas/katup/jantung 4, penyakit koroner 2,
kardiomiopati hipertrofik 1, ektopi ventrikel 1, VT 1, leukemia 1), tak diketahui 11. Irama dasar menurut
tabel: AF pada 35, 36, 37, 50 — anotasi otomatis memberi AF dominan pada 35, 36, 50 (cocok) dan hanya
14 menit pada 37. nsrdb (dokumentasi): 5 laki-laki 26–45, 13 perempuan 20–50. chfdb (header): 11
laki-laki, 4 perempuan, umur 22–71 (median 61, satu tak diketahui), NYHA III–IV.

**Kontrol pergeseran waktu dalam-pasien** (`11_daya_bootstrap.py blok`, `keluaran/daya_bootstrap_blok.log`):
blok 30 menit berurutan yang berakhir ≥ 150 menit sebelum VF sebagai "dekat palsu" lawan jendela jauh
lain (≥ 30 menit dari blok), ~14 pasien, 50 undian. Logistik AUC rerata 0,501 [0,406; 0,600], maks 0,616;
RF 0,489 [0,382; 0,592], maks 0,605. AUC nyata RF 0,608 berada di atas ke-50 undian (dekat batas);
logistik 0,560 di dalam sebaran. Pergeseran waktu saja tidak menghasilkan perbedaan sebesar itu pada RF.

**Simulasi daya dengan kohort bootstrap** (`11_daya_bootstrap.py daya`, `keluaran/daya_bootstrap.csv`,
`keluaran/daya_bootstrap_ringkas.log`; 10 Oktober 2026). Tiap simulasi mengambil ulang 12 pasien kohort
tetap (horizon), 19 pasien dalam-pasien, dan 18 subjek nsrdb dengan pengembalian (duplikat dalam satu
lipatan); regresi logistik, 5 ulangan CV horizon, 3 dalam-pasien; 100 simulasi per sel. Aturan nominal
(bootstrap subjek dua sisi; Wilcoxon satu sisi) terlalu longgar pada kohort bootstrap karena duplikat:
positif palsu di nol 0,11 (horizon), 0,14 (horizon tanpa frac_ektopik), 0,24 (dalam-pasien). Karena itu
daya **disesuaikan ukuran**: ambang dari 100 simulasi tanpa suntikan (horizon dua sisi [−0,186; +0,100];
tanpa ektopi [−0,175; +0,198]; dalam-pasien rerata selisih skor > +0,171), laju di nol 0,05–0,06.

| Perubahan pra-VF | Horizon, semua ciri | Horizon tanpa frac_ektopik | Dalam-pasien |
|---|---|---|---|
| tanpa suntikan | 0,06 | 0,06 | 0,05 |
| searah sddb − nsrdb 1 SD / 2 SD | 0,28 / 0,34 | 0,19 / 0,25 | 0,54 / 0,98 |
| tegak lurus 1 SD | 0,25 | 0,15 | 0,95 |
| denyut +10 / +20 per menit | 0,05 / 0,12 | 0,07 / 0,09 | 0,25 / 0,78 |
| ektopi naik ×2 / ×4 | 0,08 / 0,10 | (tak berubah) | 0,26 / 0,43 |
| ektopi turun ×0,5 / ×0,25 | 0,05 / 0,04 | (tak berubah) | 0,03 / 0,03 |

Selisih AUC(0) − AUC(6 jam) rerata: searah 1 SD +0,072 (sd 0,079), 2 SD +0,094; denyut +20 +0,040;
ektopi ±0,002. Bacaan:
- Dengan ketidakpastian pengambilan sampel pasien, uji horizon lawan nsrdb lemah, bukan buta: 25–34 %
  untuk perubahan HRV 1–2 SD, 12 % untuk denyut +20/menit, ≤ 10 % untuk perubahan ektopi. Sebabnya
  sebaran selisih antar-kohort (sd 0,06–0,09) dengan 12 pasien.
- **Mekanisme "frac_ektopik membawa pemisahan sehingga uji horizon buta" tidak didukung**: tanpa
  frac_ektopik di model horizon dayanya tidak naik (0,15–0,25 lawan 0,25–0,34).
- **"Uji horizon melihat turunnya ektopi" (44/50 di simulasi pertama) tidak bertahan**: turun ×0,25
  terdeteksi 4 %. Angka 44/50 berasal dari geseran semua ciri sekaligus termasuk geseran mutlak ektopi.
- Uji dalam-pasien jauh lebih peka untuk perubahan HRV (0,54–0,98) dan denyut +20 (0,78), sedang untuk
  kenaikan ektopi (0,26–0,43), tidak peka untuk turunnya ektopi (0,03), dan lemah untuk +10/menit (0,25).
- Simulasi random forest dengan kohort bootstrap tidak dijalankan (waktu); hasil RF tetap dari
  10_daya_ulang.py (kohort tetap).

**Permutasi dalam-pasien RF N = 999** (`11_daya_bootstrap.py permutasi`, `keluaran/daya_bootstrap_permutasi.log`,
2 ulangan CV): AUC asli 0,608; permutasi rerata 0,502 [0,411; 0,582]; p 0,0100 (9 dari 999 permutasi ≥ asli).
p tidak lagi di batas resolusi.

## Analisis sesudah telaah panel kelima (11 Oktober 2026; `12_analisis_akhir.py`, `keluaran/akhir.log`, `keluaran/akhir_perpasien.log`)

**Rincian per pasien kontras dalam-pasien RF** (10 ulangan CV, AUC semua 0,627; `akhir_perpasien.csv`):
selisih skor dekat − jauh positif pada 13 dari 19 pasien; AUC tinggalkan-satu-pasien 0,600–0,657 (tidak
ada satu pasien penentu). Pasien dengan jendela VT di 30 menit terakhir (6): selisih rerata +0,095 lawan
+0,071 tanpa VT. Spearman selisih skor dengan perubahan frac_ektopik (dekat − jauh) 0,67: perubahan RF
mengikuti naiknya ektopi menjelang VF.

**Kontrol pergeseran waktu sepadan** (blok 30 menit m ≥ 120 sebelum VF lawan jendela ≥ m + 120 menit, jarak 90
menit seperti kontras utama, ~15 pasien, 200 undian; `akhir_geser_*.csv`): logistik AUC rerata 0,501 [0,361;
0,626], persentil 95 0,607, maks 0,649; RF 0,511 [0,399; 0,632], persentil 95 0,612, maks 0,661. Bagian undian
≥ AUC nyata: RF 0,608 → 6 % (p empiris 0,065); logistik 0,560 → 17 % (p 0,174). **Perubahan RF dalam-pasien
tidak bisa dibedakan dari pergeseran dalam rekaman**; kontrol tak sepadan sebelumnya (jarak 30 menit, 50
undian) melebih-lebihkannya.

**Kuantisasi 128 Hz** (waktu denyut sddb dan chfdb dibulatkan ke 1/128 detik; `akhir_hz128.csv`): AUC berubah
paling banyak 0,013 (logistik lawan chfdb pada 1 jam); horizon lawan nsrdb dan dalam-pasien berubah
≤ 0,007. Beda frekuensi cuplik tidak menjelaskan hasil.

**Uji dalam-pasien pada 12 pasien kohort tetap, dua sisi** (`akhir_dalam12.csv`; sebanding dengan uji horizon:
jumlah pasien dan sisi sama; daya disesuaikan ukuran, ambang dari 100 simulasi nol):

| Perubahan pra-VF | Horizon (12, dua sisi) | Dalam-pasien (12, dua sisi) | Dalam-pasien (19, dua sisi) |
|---|---|---|---|
| tanpa suntikan | 0,06 | 0,06 | 0,06 |
| HRV searah 1 / 2 SD | 0,28 / 0,34 | 0,74 / 0,98 | 0,34 / 0,93 |
| HRV tegak lurus 1 SD | 0,25 | 0,92 | 0,90 |
| denyut +10 / +20 | 0,05 / 0,12 | 0,25 / 0,64 | 0,15 / 0,66 |
| ektopi ×2 / ×4 | 0,08 / 0,10 | 0,29 / 0,63 | 0,14 / 0,38 |
| ektopi ×0,5 / ×0,25 | 0,05 / 0,04 | 0,13 / 0,12 | 0,05 / 0,06 |

Dengan jumlah pasien dan sisi yang sama, uji dalam-pasien tetap 2,5 sampai 6 kali lebih peka. (Daya 19
pasien dua sisi untuk searah 1 SD lebih rendah daripada 12 pasien; kohort 12 berisi pasien dengan rekaman
panjang. Sebaran simulasi berbeda; jangan dibaca sebagai efek jumlah pasien.)

**Simulasi daya random forest** (12 pasien, dua sisi; 40 simulasi nol, 20 per sel lain; `akhir_rf.csv`):
selisih horizon RF tanpa suntikan rerata +0,121 (sd 0,089) — sesuai +0,116 pada data nyata, tetapi
selangnya mencakup 0 antar-kohort; dalam-pasien tanpa suntikan +0,002 (sd 0,060). Daya horizon RF 0,10–0,15
(HRV 1 dan 2 SD, denyut +20, ektopi ×4); dalam-pasien RF 0,75 / 0,90 (HRV 1 / 2 SD), 0,30 (denyut +20),
0,30 (ektopi ×4). Dengan 20 simulasi, selang binomial ±~20 poin persen.

## Analisis sesudah telaah panel keenam (11 Oktober 2026; `13_peluruhan_lambat.py`)

**Selisih daya berpasangan** (`keluaran/rasio.log`; 12 pasien, dua sisi, bootstrap 2.000 atas 100 simulasi
berpasangan): dalam-pasien − horizon = +0,46 [+0,33; +0,59] (HRV searah 1 SD), +0,64 [+0,54; +0,73] (2 SD),
+0,67 [+0,58; +0,76] (tegak 1 SD), +0,20 [+0,11; +0,29] (denyut +10), +0,52 [+0,42; +0,62] (+20), +0,21
[+0,11; +0,31] (ektopi ×2), +0,53 [+0,42; +0,64] (×4), +0,08 [0,00; +0,16] (×0,5 dan ×0,25). Rasio HRV
2,6 [1,9; 3,9], 2,9 [2,3; 3,9], 3,7 [2,7; 5,6]; rasio lain lebar (denyut +10: 5,0 [2,2; 22]).

**Peluruhan lambat** (e^(−t/180 menit); `keluaran/lambat.log`, `keluaran/lambat.csv`; ambang nol dari simulasi
nol yang ada): horizon 0,25 / 0,28 (HRV 1 / 2 SD), 0,24 (denyut +20); dalam-pasien 0,83 / 0,99 / 0,64.
Keunggulan uji dalam-pasien tidak bergantung pada peluruhan cepat.
