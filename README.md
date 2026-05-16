# 9HPT Analiza

Program za avtomatizirano analizo Testa devetih zatičev (9HPT) z uporabo računalniškega vida in strojnega učenja.

## Priprava okolja

Za zagon z Dockerjem (lokalno):
1. docker build -t matijap_rv_lokalno -f docker/Dockerfile docker/
2. docker run -it -v "$(pwd):/workspace" -w /workspace matijap_rv_lokalno bash

Testni videi morajo biti shranjeni v mapi `data/`. Izhodni posnetki in logi se shranjujejo v mapo `output/`.