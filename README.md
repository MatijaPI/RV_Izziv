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

## Uporaba CLI

Obdelava vseh videov:
python src/main.py -i all

Obdelava posameznega videa:
python src/main.py -i test.mp4

Izhod na poljubno ime/datoteko:
python src/main.py -i test.mp4 -o mojnoviizhod.mp4

## Opravljeni koraki
- **Korak 1:** Vzpostavitev strukture projekta, Docker okolja in repozitorija.
- **Korak 2:** Implementacija branja in zapisovanja videa v "headless" načinu, popravek Dockerja. Uspešen test branja in shranjevanja videa.
- **Korak 3:** Dodana integracija MediaPipe – avtomatski prenos modela in zaznavanje ter izris sklepov in povezav roke v vsak okvir izhodnega videa.
- **Korak 4:** Podpora za ukazno vrstico (`--input`, `--output`) ter procesiranje posameznega ali vseh videov v strukturi data/.
- **Korak 5:** Zapisovanje dnevnika procesiranja za vsak video v mapo `output/logs/` (ime loga sledi izhodnemu videu), kar omogoča sledljivost ter enostavno preverjanje uspešne obdelave.
- **Korak 6:** Izračun in izvoz časovnih vrst poti, hitrosti in pospeška zapestja v CSV.
- **Korak 7:** Real-time prikaz poti, hitrosti in pospeška z mini grafi v overlay-ju na izhodnem videu.  