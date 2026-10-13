# U1 — Kubernetes klasteris ir MLflow serveris

Individuali užduotis. Sudaro **mini MLOps platformos** pirmąją pusę: atkartojamas pipeline’as ir savas MLflow serveris lokaliame Kubernetes klasteryje.

| | |
|---|---|
| **Pateikimas** | **2026-10-21** (8 savaitė) |
| **Gynimas** | gyvai per laboratorinį darbą tą pačią savaitę (5–10 min.); prireikus — vėlesniame užsiėmime |
| **Tag** | `u1-final` |
| **Balai** | U1 + U2 kartu iki **3 balų** (30 % kurso). Egzaminui reikia U1 + U2 **≥ 1,5** |
| **Medžiaga** | branduolys — 1–6 sav. (CI/CD perkeltas į U2; 7–8 sav. nereikalinga) |

---

## Tikslas

Turėti savo infrastruktūrą, į kurios MLflow serverį **klasteryje** log’inami eksperimentai. Po U1 turite gebėti paleisti pipeline’ą, atkurti istorinį run’ą ir parodyti rezultatus klasterio UI.

## Terminai

| Sav. | Data | Ką turėti |
|---|---|---|
| 1 | 2026-09-02 | Savo GitHub repo; modelio ir duomenų pasirinkimas; U1 paskelbta |
| 2 | — | DVC + MLflow lokaliai; `docs/requirements.md` v1.1 kryptis |
| 5 | 2026-09-30 | Lokalus klasteris laboratorijoje (`kind` referencinis) |
| 6 | 2026-10-07 | Paskutinė branduoliui reikalinga paskaita |
| **8** | **2026-10-21** | **Pateikimas (`u1-final`) ir gynimas** |

Pateikimas = tag `u1-final` **iki datos**. Be gyvos demonstracijos užduotis **nevertinama**.

## Bendros taisyklės

1. **Savo repo.** Kiekvienam studentui atskira GitHub repozitorija (struktūra pagal `tutorial/fraud-detection`). `docs/requirements.md` versijuojamas (v1.0 → v1.1).
2. **Savo modelis ir duomenys.** Kurso fraud detection pavyzdys **neleidžiamas**. Rekomenduojami atviri Hugging Face rinkiniai. Užtenka mažo tabulinio / tekstinio modelio (mokymas CPU per kelias minutes) — vertinamas **srautas aplink modelį**, ne jo SOTA kokybė.
3. **Savo MLflow.** Bendro kurso serverio nėra. U1 hostinate Tracking Server **savo** klasteryje ir juo naudositės U2 bei komandiniame projekte.
4. **Aplinka.** Docker + lokalus Kubernetes (žr. [Klasteris](#lokalus-kubernetes-klasteris)). GPU **neprivalomas**; nenaudojant GPU balas nemažinamas. GCP kreditai — papildoma galimybė.
5. **Kodas ir DI asistentai.** Leidžiama naudoti `tutorial/fraud-detection`, lab. medžiagą ir DI asistentus, bet gynime privalote **paaiškinti ir pakeisti** kiekvieną pateiktą failą. Kitaip balas mažinamas arba darbas neužskaitomas.
6. **Branduolys vs. išplėstinė dalis.** Branduolys būtinas užskaitymui. E1–E5 — papildomi balai iki U1 maksimumo.
7. **README** — kaip paleisti nuo nulio, portai, prisijungimai, kur žiūrėti rezultatus.

## Reikalavimai (branduolys)

### U1-R1 — Repo ir reikalavimai

- Katalogai: `src/`, `docker/`, `k8s/`, `docs/`.
- README (paleidimas nuo nulio).
- `docs/requirements.md` **≥ v1.1**: ≥3 FR, ≥5 NFR su KPI, tarp jų **reprodukcijos NFR**, savo pasirinktai problemai.

### U1-R2 — Atkartojamas pipeline’as

- `dvc.yaml` su ≥3 stage’ais: `prepare` → `train` → `evaluate`.
- `params.yaml`; `dvc.lock` **commit’intas**.
- DVC remote: lokalus katalogas, SSH, MinIO arba GCS.
- ≥3 MLflow run’ai su skirtingais hiperparametrais.
- `reproduce` scenarijus atkuria pasirinktą run’ą per **< 2 min.**

### U1-R3 — Docker vaizdas

- `Dockerfile` repozitorijoje (mokymo arba inferencijos).
- Publikuotas GHCR su tag’u `sha-<commit>` (ne `latest`).
- Vaizdo dydis ir Trivy skenavimo rezultatas užfiksuoti `docs/`.

### U1-R4 — Klasteris ir MLflow

- Lokalus Kubernetes; konfigūracija repo (pvz. `k8s/kind-config.yaml`).
- Atskiras Namespace.
- **MLflow Tracking Server**: Deployment + Service + PVC (rekomenduojama PostgreSQL + artefaktų PVC).
- Pasiekiamas iš kompiuterio (port-forward arba NodePort).
- Mokymo scenarijus log’ina **į klasterio** MLflow, ne tik į lokalų `mlruns/`.

## Išplėstinė dalis (neprivaloma)

| ID | Kas |
|---|---|
| **U1-E1** | MLflow su autentifikacija (`--app-name basic-auth`), ≥2 vartotojai / rolės |
| **U1-E2** | Mokymas kaip Kubernetes Job, log’inantis į klasterio MLflow |
| **U1-E3** | Klasteris kaip kodas: Terraform (kind / minikube / k3d) + Helm arba Kustomize MLflow serveriui |
| **U1-E4** | DVC remote debesyje (GCS per Terraform) |
| **U1-E5** | GPU lokaliame klasteryje: `nvidia.com/gpu` ir Pod su `nvidia-smi` |

## Deliverable’ai

| ID | Ką pateikti |
|---|---|
| **D1** | Repo URL + tag `u1-final` |
| **D2** | `docs/requirements.md` ≥ v1.1 |
| **D3** | GHCR vaizdo nuoroda |
| **D4** | `k8s/` manifestai + klasterio konfigūracija |
| **D5** | MLflow UI su ≥3 run’ais **klasterio** serveryje (demonstracija) |
| **D6** | `docs/u1-report.md` (1 psl.: kas padaryta, kas neveikė, kiek laiko užtruko) |

## Vertinimas

| Kriterijus | Minimalus slenkstis |
|---|---|
| Repo, README, `requirements.md` su KPI | ≥5 NFR su KPI |
| MLflow + DVC, `reproduce` < 2 min. | `dvc repro` veikia; ≥3 run’ai |
| Docker vaizdas GHCR, skenuotas | vaizdas pasileidžia |
| Lokalus klasteris + MLflow jame | run’as iš kompiuterio matomas klasterio UI |
| Gynimas (demo, klausimai, live edit) | atsako į dėstytojo klausimus |
| Išplėstinė dalis (E1–E5) | neviršijant U1 maksimumo |

Konkretūs balai pagal kriterijus — dėstytojo rubrikoje. Savaitiniai laboratoriniai darbai (iki 0,2 / sav.) yra **papildomi** balai, ne U1 pakaitalas.

## Lūkesčiai

**Tikimasi**

- Sistema paleidžiama iš README be „paslėptų“ žingsnių.
- Duomenys, parametrai ir run’ai atsekami (Git + DVC + MLflow).
- MLflow artefaktai **išlieka** perkūrus Pod’ą (PVC).
- Tag’ai versijuoja vaizdus (`sha-<commit>`), ne `latest`.
- Gynime galite pakeisti parametrą / NFR ir paaiškinti, kas persileis `dvc repro`.

**Nėra tikimasi**

- Geriausio modelio ar GPU.
- Pilno CI/CD (tai U2).
- Serving’o / inferencijos API (tai U2).
- Komandinio projekto temos — U1 yra **individuali**.

**Dažnos klaidos**

- Run’ai tik lokaliame `mlruns/`, ne klasterio serveryje.
- `dvc.lock` necommit’intas; `reproduce` > 2 min. arba atkuria kitą run’ą.
- GHCR tag `latest`; nėra Trivy rezultato.
- `requirements.md` be KPI arba be reprodukcijos NFR.
- README neleidžia paleisti nuo nulio.

## Gynimo klausimų pavyzdžiai

- Kur fiziškai yra MLflow artefaktai klasteryje ir kas nutiks perkūrus Pod’ą?
- Kas pasikeis `dvc.lock`, pakeitus `params.yaml`?
- Kodėl tag’as `latest` — bloga praktika?
- Paaiškinkite kiekvieną `kubectl get all -n <ns>` objektą.

## Lokalus Kubernetes klasteris

Klasteris veikia kaip Docker konteineris (-iai) jūsų kompiuteryje ir aprašomas kaip kodas.

| Įrankis | Konfigūracija kaip kodas | GPU |
|---|---|---|
| **kind** (referencinis) | `kind-config.yaml` | `nvkind` + device plugin |
| **minikube** (docker driver) | `minikube start` flag’ai | `--gpus all` (nuo v1.32) — paprasčiausias GPU kelias |
| **k3d** | `k3d-config.yaml` | savas k3s + CUDA vaizdas |
| Docker Desktop Kubernetes | nėra | nerekomenduojama |

**Rekomendacija:** branduolys — CPU (`kind`). Ištekliai: MLflow + PostgreSQL ≈ 4 GB RAM; silpnesniam kompiuteriui — GCP VM su tuo pačiu `kind`.

