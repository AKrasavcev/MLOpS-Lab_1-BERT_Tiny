# Ataskaita

## Kas padaryta

* **U1-R1** - Padaryta
* **U1-R2** - Padaryta
* **U1-R3** - Padaryta
* **U1-R4** - Padaryta

Išplestinė dalis (**E1-E5**) nebuvo implementuota.

Implementuotas modelio ir mokymo duomenų parsiuntimas iš **HuggingFace** ir tolesnis fine-tuninimas su **CPU**. Visa mokymo run'ų informacija *(parametrai, train ir eval metrikos, modelis)* yra saugoma **MLflow** serveryje ir lokaliai `model/` kataloge (išskyrus parametrus (`params.yaml`), jie saugomi projekto ROOT direktorijoje). **MLflow Tracking Server** yra paleistas **Minikube** klasteryje ir yra susietas su **PostgreSQL** serveriu, kuris taip pat paleistas klasteryje. 

Daugiau laiko negu tikėjausi užtruko klasterio konfiguracija ir pod'ų paleidimas.

Visą darbą atlikti užtruko apie 1 savaitę

Katalogai:

* `data/` mokymo duomenys
* `docker/` Dockerfile'as 
* `docs/` dokumentacija & **Trivy** skenavimo rezultatai
* `k8s/` **Kubernetes** manifestai 
* `model/` saugomi paskutinio *run'o* metrikos ir ID
* `scripts/` padalbiniai scriptai
* `src/` pagrindiniai kodo failai

