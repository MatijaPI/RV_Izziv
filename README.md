# 9HPT Analiza

Program za avtomatizirano analizo Testa devetih zatičev (9HPT) z uporabo računalniškega vida in strojnega učenja.

## Priprava okolja 

**Lokalni zagon (Mac/PC):**
```bash
docker build -t matijap_rv_lokalno -f docker/Dockerfile docker
docker run -v "$(pwd):/workspace" -w /workspace matijap_rv_lokalno python src/main.py --input all
```

**Zagon na strežniku:**
```bash
docker build -t matijap_rv_server -f docker/Dockerfile docker/
docker run --shm-size=16g -it \
  -v /media/FastDataMama/matijap/:/workspace \
  -v /media/FastDataMama/data_rv_26/:/data \
  -w /workspace matijap_rv_server bash
```

Testni videi morajo biti shranjeni v mapi `data/`. Izhodni posnetki, logi in grafi se shranjujejo v `output/`.

---

## Kalibracija kamer

Program podpira kalibracijo treh kamer za natančno preračunavanje kinematičnih parametrov v metrične enote (mm, mm/s, mm/s²).

### Struktura kalibracijskih datotek

```
calibration/
├── left/              # Kalibracijske slike za levo kamero (camP_0)
├── mid/               # Kalibracijske slike za sredinsko kamero (camP_1)
├── right/             # Kalibracijske slike za desno kamero (camP_2)
├── scripts/
│   └── calibrate.py   # Kalibracijska skripta
├── conf/              # Izhodne konfiguracijske datoteke (generirane)
│   ├── left_calibration.json
│   ├── mid_calibration.json
│   └── right_calibration.json
└── calibration.json   # Validacijska/fallback datoteka
```

### Izvedba kalibracije

1. Dodajte kalibracijske slike (šahovnica 9x6, stranica 20mm) v ustrezne mape:
   - `calibration/left/` za levo kamero (camP_0)
   - `calibration/mid/` za sredinsko kamero (camP_1)  
   - `calibration/right/` za desno kamero (camP_2)

2. Zaženite kalibracijo:
   ```bash
   python calibration/scripts/calibrate.py
   ```
3. Konfiguracijske datoteke se shranijo v `calibration/conf/`.

### Kako deluje

- Program iz imena vhodne datoteke avtomatsko prepozna kamero (`camP_0` → left, `camP_1` → mid, `camP_2` → right).
- Naloži ustrezno kalibracijsko konfiguracijo in odstrani distorzijo slike.
- Preračuna kinematične parametre iz pikslov v milimetre.
- Če kalibracija ni na voljo, program deluje v pikselskem načinu (nazaj združljivo).
- Fallback: če `calibration/conf/` ne obstaja, se uporabi `calibration/calibration.json`.

---

## Uporaba CLI

### Osnovni primeri

```bash
# Obdelava vseh videov v data/
python src/main.py --input all

# Obdelava enega videa
python src/main.py --input test.mp4

# Z ročno določenim izhodnim imenom
python src/main.py --input test.mp4 --output rezultat.mp4

# Brez kalibracije (pikselske enote)
python src/main.py --input test.mp4 --no-calibration

# Brez prikaza ROI pravokotnika
python src/main.py --input test.mp4 --no-roi

# Ročno določen ROI (kot deleži slike)
python src/main.py --input test.mp4 --roi 0.2 0.1 0.8 0.9

# Glajenje položajev z večjo sigmo (bolj gladke krivulje)
python src/main.py --input test.mp4 --smooth-sigma 3.0

# Glajenje izklopljeno
python src/main.py --input test.mp4 --smooth-sigma 0.0

# Savitzky-Golay post-process filter za grafe (zahteva scipy)
python src/main.py --input test.mp4 --smooth-method savgol

# Prilagoditev praga zaznave prijema zatiča
python src/main.py --input test.mp4 --pinch-thr-mm 15.0
```

---

## Opis vseh argumentov

| Argument | Kratica | Privzeto | Opis |
|---|---|---|---|
| `--input` | `-i` | — | **Obvezen.** Pot do vhodne `.mp4` datoteke ali `all` za obdelavo vseh videov v mapi `data/`. |
| `--output` | `-o` | samodejno | Pot/ime izhodne `.mp4` datoteke. Uporablja se samo pri obdelavi enega videa. Če ni podano, se doda pripona `_obdelan`. |
| `--no-calibration` | — | izklopljeno | Onemogoči uporabo kalibracijskih datotek. Vsi kinematični parametri se izračunajo v pikslih (px, px/s, px/s²). |
| `--no-roi` | — | izklopljeno | Skrije ROI (Region of Interest) pravokotnik na izhodnem videu. |
| `--roi X1 Y1 X2 Y2` | — | per kamera | Ročno nastavi ROI kot deleže slike (vrednosti med 0.0 in 1.0). Primer: `--roi 0.2 0.1 0.8 0.9`. Prepiše privzete ROI vrednosti za vse kamere. |
| `--lock-after N` | — | `30` | Število okvirjev po katerih se zaklene izbira aktivne roke. Do tega trenutka se aktivna roka določa dinamično glede na gibanje. |
| `--smooth-sigma σ` | — | `1.5` | Standardna deviacija Gaussovega glajenja položajev v okvirjih. Glajenje deluje kavzalno v realnem času (brez zakasnitve). Vrednost `0.0` glajenje izklopi. Priporočene vrednosti: `1.0`–`3.0`. |
| `--smooth-method` | — | `gauss` | Metoda post-process glajenja za izvozne grafe in CSV. Možnosti: `gauss` (Gaussova konvolucija, brez dodatnih odvisnosti) ali `savgol` (Savitzky-Golay filter, zahteva `scipy`). |
| `--birds-eye` | — | izklopljeno | Vstavi bird's-eye view vstavek v spodnji desni kot izhodnega videa. Zahteva homografijsko matriko iz kalibracije. |
| `--bev-size W H` | — | `600 600` | Velikost bird's-eye platna v pikslih. |
| `--bev-scale S` | — | `2.0` | Merilo bird's-eye pogleda v px/mm. |
| `--pinch-thr-mm D` | — | `20.0` | Prag razdalje med konico palca in kazalca za zaznavo prijema zatiča v milimetrih (pri kalibriranem načinu). |
| `--pinch-thr-px D` | — | `40.0` | Prag razdalje med konico palca in kazalca za zaznavo prijema zatiča v pikslih (brez kalibracije). |

---

## Izhodni podatki

Za vsak obdelan video se generirajo:

| Datoteka | Mapa | Opis |
|---|---|---|
| `*_obdelan.mp4` | `output/videos/` | Video z izrisanim skeletom roke, oznakami palca/kazalca in HUD pasom s kinematičnimi vrednostmi. |
| `*_kinematika.csv` | `output/logs/` | Časovne vrste poti, hitrosti in pospeška za zapestje, palec in kazalec ter pinch razdaljo. Vrednosti so post-process glajene. |
| `*.log` | `output/logs/` | Dnevnik obdelave: kamera, kalibracija, ROI, zaklep roke, dogodki prijema/odlaganja, nastavitve glajenja. |
| `*_graf.png` | `output/graphs/` | Graf s 4 subplot-i: pot, hitrost, pospešek (surovo + glajeno) in pinch razdalja z oznakami prijemov. |

---

## Opravljeni koraki
- **Korak 1:** Vzpostavitev strukture projekta, Docker okolja in repozitorija.
- **Korak 2:** Implementacija branja in zapisovanja videa v "headless" načinu, popravek Dockerja. Uspešen test branja in shranjevanja videa.
- **Korak 3:** Dodana integracija MediaPipe – avtomatski prenos modela in zaznavanje ter izris sklepov in povezav roke v vsak okvir izhodnega videa.
- **Korak 4:** Podpora za ukazno vrstico (`--input`, `--output`) ter procesiranje posameznega ali vseh videov v strukturi data/.
- **Korak 5:** Zapisovanje dnevnika procesiranja za vsak video v mapo `output/logs/` (ime loga sledi izhodnemu videu), kar omogoča sledljivost ter enostavno preverjanje uspešne obdelave.
- **Korak 6:** Izračun in izvoz časovnih vrst poti, hitrosti in pospeška zapestja v CSV.
- **Korak 7:** Real-time prikaz poti, hitrosti in pospeška z mini grafi v overlay-ju na izhodnem videu.
- **Korak 8:** Kalibracija kamer – podpora za tri kamere (left/mid/right), avtomatska prepoznava kamere iz imena datoteke, odstranitev distorzije, pretvorba kinematičnih parametrov v metrične enote (mm). Validacija s primerjavo z referenčno kalibracijo (max. 2% odstopanje).
- **Korak 9:** Selekcija aktivne roke – ROI filter in `ActiveHandSelector` z zaklepom po N okvirjih za robustno sledenje v primerih, ko je v vidnem polju več rok (aktivna + neaktivna roka).
- **Korak 10:** Kinematika palca in kazalca – ločeno sledenje konicama palca (landmark 4) in kazalca (landmark 8) z izračunom d/v/a za vsako točko posebej.
- **Korak 11:** Zaznava prijema in odlaganja zatiča (`PinchDetector`) na podlagi razdalje palec–kazalec s pragom in potrditvijo v N zaporednih okvirjih.
- **Korak 12:** Izvoz kinematičnih grafov v `output/graphs/` (matplotlib PNG, 4 subplot-i: pot, hitrost, pospešek, pinch razdalja).
- **Korak 13:** Nov HUD dizajn – horizontalni pas na dnu videa s tremi bloki (Zapestje / Palec / Kazalec) in diskretnim pinch indikatorjem.
- **Korak 14:** Gaussovo glajenje položajev – kavzalni `KinematicSmoother` za zanesljivejše določanje hitrosti in pospeškov brez zakasnitve v videu; post-process glajenje (Gauss/Savitzky-Golay) za izvozne grafe in CSV.