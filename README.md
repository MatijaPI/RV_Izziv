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

## Opravljeni koraki
- **Korak 1:** Vzpostavitev strukture projekta, Docker okolja in repozitorija.
- **Korak 2:** Implementacija branja in zapisovanja videa v "headless" načinu, popravek Dockerja. Uspešen test branja in shranjevanja videa.
- **Faza 3:** Dodana integracija MediaPipe – avtomatski prenos modela in zaznavanje ter izris sklepov in povezav roke v vsak okvir izhodnega videa.