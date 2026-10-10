# Hasil Uji CHF — artikel EKG 3 (calon)

Dijalankan 27 September 2026. Skrip `01_jendela_chf.py` dan `02_uji_chf.py`; tabel
`keluaran/hasil_chf.csv` (tidak masuk repo).

**Data.** 32.004 jendela HRV 5 menit dari 156 subjek, lima basis data PhysioNet terbuka: chfdb (15,
250 Hz, usia rerata 56), chf2db (29, 128 Hz, 55), nsrdb (18, 128 Hz, 34), nsr2db (54, 128 Hz, 62),
fantasia (20 muda 21-34 th + 20 tua 68-85 th, 250 Hz, rekaman 2 jam berbaring). Ciri sama dengan
artikel 2. Skor subjek dari maksimal 6 jendela; pembagian subjek, 5 lipatan x 10 ulangan.

## Temuan — berlawanan dengan dugaan

| Uji | Logistik AUC [95 %] | RF AUC [95 %] |
|---|---|---|
| T1 rancangan baku: CHF (chfdb + chf2db) lawan sehat (nsrdb + nsr2db) | 0,940 [0,88, 0,98] | 0,925 [0,86, 0,98] |
| T3 latih chfdb + nsrdb, uji chf2db + nsr2db | **0,895** [0,81, 0,97] | **0,881** [0,78, 0,96] |
| T3 latih chf2db + nsr2db, uji chfdb + nsrdb | **0,981** [0,93, 1,00] | **0,981** [0,93, 1,00] |
| T2 probe: chfdb lawan chf2db (sama-sama CHF, 250 lawan 128 Hz) | **0,522** [0,33, 0,71] | **0,485** [0,32, 0,66] |
| T2 probe: nsrdb lawan nsr2db (sama-sama sehat, usia 34 lawan 62) | 0,786 [0,65, 0,91] | 0,813 [0,68, 0,92] |
| T2 probe: fantasia muda lawan nsrdb (sama-sama sehat dan muda) | 0,753 [0,58, 0,90] | 0,756 [0,59, 0,91] |

T4, model T1 diterapkan pada Fantasia: orang sehat tua dinyatakan CHF **10 %** (kedua model), muda
5 %. AUC skor-CHF tua lawan muda 0,725 (logistik) dan 0,810 (RF).

## Tafsiran

**Deteksi gagal jantung dari HRV pada data PhysioNet tampaknya nyata, tidak sekadar rancu basis
data** — kebalikan dari artikel 1 (SCD) dan artikel 2 (AF):

1. Model **berpindah antarbasis data** dengan AUC 0,88-0,98, termasuk dari 250 Hz ke 128 Hz.
2. Model **tidak bisa membedakan dua basis data CHF** (0,49-0,52), padahal frekuensi cuplik dan
   rumah sakitnya berbeda: ciri HRV tidak membawa sidik basis data di kelas ini.
3. Orang sehat tua **tidak** dinyatakan CHF (10 %), jadi model bukan sekadar pendeteksi usia —
   meski skornya naik dengan usia (AUC tua lawan muda 0,73-0,81).

Yang tetap perlu dicatat:

- Dua basis data sehat **dapat** dibedakan (0,79-0,81). Penjelasan paling mungkin usia (34 lawan 62
  th). Fantasia muda lawan nsrdb (0,75) menunjukkan kondisi rekaman juga terbaca: Fantasia
  berbaring 2 jam, nsrdb Holter 24 jam berjalan.
- Arah T3 tidak simetris (0,89 lawan 0,98): chf2db memuat NYHA I-III yang lebih ringan, jadi lebih
  sulit dideteksi daripada chfdb (NYHA III-IV).

## Akibatnya untuk artikel 3

Gerbang batal-lanjut di rencana artikel 2 berlaku juga di sini: dugaan "tolok ukur ini mengukur
basis data" **salah untuk CHF**. Itu tidak mematikan pekerjaannya, tetapi mengubah bentuknya:

- Sebagai artikel tersendiri, pesannya "deteksi CHF dari HRV lolos uji antarbasis data" — hasil
  positif, berguna, tetapi kurang tajam.
- Nilainya yang lebih besar: **protokol uji yang sama membedakan tolok ukur yang rancu (SCD, AF) dari
  yang tidak (CHF)**. Protokol yang selalu menjawab "rancu" tidak berguna; yang ini tidak. Itu
  bukti validitas protokolnya sendiri.

Pilihan bentuk diserahkan ke pengguna — lihat ringkasan di percakapan 27 September 2026.
