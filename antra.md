# U2 — Modelio mokymas su MLflow ir hostingas klasteryje

Individuali užduotis. Sudaro **mini MLOps platformos** antrąją pusę: nuo duomenų iki veikiančio inferencijos API **tame pačiame** U1 klasteryje, su registru, kokybės vartu ir rollback.

| | |
|---|---|
| **Paskelbimas** | 2026-10-14 (7 savaitė) |
| **Pateikimas** | **2026-11-11** (11 savaitė) |
| **Gynimas** | gyvai per laboratorinį darbą tą pačią savaitę (5–10 min.); prireikus — vėlesniame užsiėmime |
| **Tag** | `u2-final` |
| **Balai** | U1 + U2 kartu iki **3 balų** (30 % kurso). Egzaminui reikia U1 + U2 **≥ 1,5** |
| **Priklausomybė** | U1 infrastruktūra (klasteris + savas MLflow serveris) |
| **Medžiaga** | branduolys — 2, 5, 6, 7 sav.; 8 sav. (GPU) ir 9 sav. (orkestravimas) — tik išplėstinė dalis |

---

## Tikslas

Pilnas kelias: duomenys → mokymas su pilnu tracking’u → registras (`@champion`) → mokymo Job ir inferencijos servisas klasteryje → CI/CD su kokybės vartu ir rollback.

## Terminai

| Sav. | Data | Ką turėti |
|---|---|---|
| 7 | 2026-10-14 | U2 paskelbta; U1 jau apginta arba beveik paruošta |
| 8 | 2026-10-21 | U1 pateikimas; konsultacija GPU/HPC, CI/CD |
| **11** | **2026-11-11** | **U2 pateikimas (`u2-final`) ir gynimas** |

Pateikimas = tag `u2-final` **iki datos**. Be gyvos demonstracijos užduotis **nevertinama**.

## Bendros taisyklės

1. Tas pats **individualus** GitHub repo kaip U1. Tag `u2-final`; `docs/requirements.md` ≥ **v1.6**.
2. **Savo** modelis ir duomenys; kurso fraud detection **neleidžiamas**. Vertinamas srautas, ne modelio kokybė.
3. Mokymas ir serving log’ina / skaito **U1 klasterio MLflow**, ne atsitiktinį lokalų serverį.
4. GPU **neprivalomas**. E1–E5 rekomenduojami, bet balas **nemažinamas**, jei išplėstinė dalis nedaroma.
5. Referencinis kodas ir DI asistentai leidžiami; gynime privalote paaiškinti ir modifikuoti failus.
6. README atnaujintas: kaip paleisti mokymą, Job, API, kur žiūrėti MLflow ir metrikas.

## Reikalavimai (branduolys)

### U2-R1 — Mokymas „pagal visas taisykles“

MLflow run’e fiksuojama:

- parametrai;
- metrikos per žingsnius (`step`);
- sistemos metrikos;
- duomenų rinkinys (`log_input` su digest’u);
- seed’as ir git commit žymos;
- aplinka prie modelio;
- **įdėtiniai run’ai** hiperparametrų paieškai: ≥1 paieška su **≥5 bandymais** (Optuna, Hydra `--multirun` arba sklearn autolog).

### U2-R2 — Modelis registre

- `log_model(name=…)`, registruotas Model Registry su alias’u `@champion`.
- Inferencijos servisas ima modelį **iš registro pagal alias’ą**, ne iš failo diske.

### U2-R3 — Mokymo Job klasteryje

- Kubernetes Job (CPU), log’inantis į U1 MLflow.
- Parametrai per ConfigMap, prisijungimai per Secret (ne Git’e).
- Job baigiasi `Completed`; run’as matomas UI.

### U2-R4 — Inferencijos API

- FastAPI / `mlflow models serve` / KServe konteineryje.
- Deployment + Service; `/health` ir `/predict`.
- Bazinis apkrovos testas (pvz. 100 užklausų); **p95 vėlinimas** užfiksuotas `docs/`.

### U2-R5 — CI

GitHub Actions: lint + unit testai + vaizdo build ir publikavimas į GHCR.

### U2-R6 — CD su kokybės vartu ir rollback

Grandine: testai → `dvc repro` (arba mokymo Job) → kokybės vartas (metrika ≥ slenkstis iš `metrics.json`) → vaizdo build/push → manifesto atnaujinimas.

Privaloma parodyti:

- **žalią** paleidimą (deploy įvyksta);
- **raudoną** paleidimą (blogas modelis **nepraeina**);
- rollback į ankstesnę versiją.

### U2-R7 — Reikalavimai su matavimais

`docs/requirements.md` ≥ v1.6 su inferencijos NFR (p95 vėlinimas, pralaidumas, modelio dydis) ir **išmatuotomis** reikšmėmis, ne tik planais.

## Išplėstinė dalis (neprivaloma)

Balas nemažinamas, jei nedaroma.

| ID | Kas |
|---|---|
| **U2-E1** | GPU mokymas lokaliame klasteryje arba HPC (AMP, profiliavimas); HPC run’ai perkeliami į klasterio MLflow (`mlflow-export-import`) |
| **U2-E2** | Orkestravimas: prepare → train → eval → register → deploy kaip Ray / Prefect / Airflow DAG su sąlyginiu žingsniu ir HITL (Prefect — lengviausias išteklių prasme) |
| **U2-E3** | HPA inferencijos Deployment’ui + Locust apkrovos testas |
| **U2-E4** | ONNX eksportas, mažesnis inferencijos vaizdas, vėlinimo palyginimas |
| **U2-E5** | ArgoCD / canary diegimas |

## Deliverable’ai

| ID | Ką pateikti |
|---|---|
| **D1** | Tag `u2-final` |
| **D2** | MLflow UI: HPO run’ų medis, registruotas modelis su alias’u |
| **D3** | `k8s/` Job + Deployment + Service manifestai |
| **D4** | CI/CD workflow su kokybės vartu (žalias **ir** raudonas paleidimas) |
| **D5** | `docs/u2-report.md` (≤ 2 psl.: architektūra, matavimai, rollback įrodymas) |
| **D6** | `docs/requirements.md` ≥ v1.6 |

## Vertinimas

| Kriterijus | Minimalus slenkstis |
|---|---|
| Mokymas su pilnu tracking’u + HPO | ≥5 bandymai; sistemos metrikos matomos |
| Registras + alias; servisas ima modelį iš registro | `models:/<name>@champion` |
| Mokymo Job klasteryje | Job `Completed`; run’as UI |
| Inferencijos servisas + matavimai | `/predict` atsako; p95 užfiksuotas |
| CI (lint, testai, build/push) | workflow žalias |
| CD: kokybės vartas + rollback | raudonas paleidimas blokuoja deploy |
| Gynimas | live edit: pakeisti slenkstį / parametrą |
| Išplėstinė dalis (E1–E5) | neviršijant U2 maksimumo |

Konkretūs balai — dėstytojo rubrikoje. Savaitiniai lab. darbai (iki 0,2 / sav.) — papildomi balai.

## Lūkesčiai

**Tikimasi**

- Vienas nuoseklus kelias nuo commit’o iki `/predict`, be rankinio „nukopijuoti pkl į Pod’ą“.
- Alias `@champion` yra vienintelis serving’o šaltinis.
- Kokybės vartas **tikrai** blokuoja blogą modelį (raudonas CI), o ne tik aprašytas README.
- Rollback parodytas, ne tik pažadėtas.
- NFR lentelėje yra skaičiai iš jūsų matavimų.
- Secret’ai ne Git istorijoje.

**Nėra tikimasi**

- LLM 7B, vLLM ar GPU (tai išplėstinė / vėlesnės savaitės).
- Pilno komandinio stebėsenos stack’o (Prometheus/Grafana — komandinė užduotis).
- Naujo klasterio — naudojate U1.

**Dažnos klaidos**

- Servisas skaito `models/model.pkl` iš atvaizdo, ignoruodamas registrą.
- HPO be įdėtinių run’ų (penki atskiri rankiniai paleidimai nelaikomi paieška).
- Tik žalias CI; raudono scenarijaus nėra.
- p95 „apytiksliai 100 ms“ be testo.
- Job naudoja hardcodintą MLflow URI ir prisijungimus YAML’e.

## Gynimo lūkesčiai

Live edit pavyzdžiai:

- pakeisti kokybės slenkstį ir parodyti, kad deploy blokuojamas;
- pakeisti hiperparametrą ConfigMap’e ir paleisti Job;
- paaiškinti, kaip servisas randa `@champion`.
