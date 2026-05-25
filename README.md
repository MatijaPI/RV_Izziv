# 9HPT Analiza

Program za avtomatizirano analizo Testa devetih zatičev (9HPT) z uporabo računalniškega vida in strojnega učenja.

## Priprava okolja 

**Lokalni zagon (Mac/PC):**
Za zagon z Dockerjem (lokalno):
1. docker build -t matijap_rv_lokalno -f docker/Dockerfile docker
2. docker run -v "$(pwd):/workspace" -w /workspace matijap_rv_lokalno python src/main.py

**Zagon na strežniku**
1. docker build -t matijap_rv_server -f docker/Dockerfile docker/
2. docker run --shm-size=16g -it -v /media/FastDataMama/matijap/:/workspace -v /media/FastDataMama/data_rv_26/:/data -w /workspace matijap_rv_server bash

Testni videi morajo biti shranjeni v mapi `data/`. Izhodni posnetki in logi se shranjujejo v mapo `output/`.

## Kalibracija kamer

Program podpira kalibracijo treh kamer za natančno preračunavanje kinematičnih parametrov v metrične enote (mm, mm/s, mm/s²).

### Struktura kalibracijskih datotek

```
calibration/
├── left/              # Kalibracijske slike za levo kamero (camP_0)
│   └── calibration_0.jpg
├── mid/               # Kalibracijske slike za sredinsko kamero (camP_1)
│   └── calibration_1.jpg
├── right/             # Kalibracijske slike za desno kamero (camP_2)
│   └── calibration_2.jpg
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

- Program iz imena vhodne datoteke avtomatsko prepozna kamero:
  - `camP_0` → leva kamera
  - `camP_1` → sredinska kamera
  - `camP_2` → desna kamera
- Na podlagi prepoznane kamere naloži ustrezno kalibracijsko konfiguracijo
- Odstrani distorzijo slike (lens undistortion)
- Preračuna kinematične parametre (pot, hitrost, pospešek) iz pikslov v milimetre
- Če kalibracija ni na voljo, program deluje v pikselskem načinu (nazaj združljivo)

### Fallback mehanizem

Če lastna kalibracija (`calibration/conf/`) ne obstaja, program uporabi validacijsko datoteko `calibration/calibration.json` kot fallback.

## Uporaba CLI

Obdelava vseh videov:
```bash
python src/main.py -i all
```

Obdelava posameznega videa:
```bash
python src/main.py -i test.mp4
```

Izhod na poljubno ime/datoteko:
```bash
python src/main.py -i test.mp4 -o mojnoviizhod.mp4
```

Onemogočanje kalibracije:
```bash
python src/main.py -i test.mp4 --no-calibration
```

## Opravljeni koraki
- **Korak 1:** Vzpostavitev strukture projekta, Docker okolja in repozitorija.
- **Korak 2:** Implementacija branja in zapisovanja videa v "headless" načinu, popravek Dockerja. Uspešen test branja in shranjevanja videa.
- **Korak 3:** Dodana integracija MediaPipe – avtomatski prenos modela in zaznavanje ter izris sklepov in povezav roke v vsak okvir izhodnega videa.
- **Korak 4:** Podpora za ukazno vrstico (`--input`, `--output`) ter procesiranje posameznega ali vseh videov v strukturi data/.
- **Korak 5:** Zapisovanje dnevnika procesiranja za vsak video v mapo `output/logs/` (ime loga sledi izhodnemu videu), kar omogoča sledljivost ter enostavno preverjanje uspešne obdelave.
- **Korak 6:** Izračun in izvoz časovnih vrst poti, hitrosti in pospeška zapestja v CSV.
- **Korak 7:** Real-time prikaz poti, hitrosti in pospeška z mini grafi v overlay-ju na izhodnem videu.
- **Korak 8:** Kalibracija kamer – podpora za tri kamere (left/mid/right), avtomatska prepoznava kamere iz imena datoteke, odstranitev distorzije, pretvorba kinematičnih parametrov v metrične enote (mm).