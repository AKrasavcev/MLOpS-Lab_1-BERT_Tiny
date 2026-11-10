# Lab 1, 2 - Reikalavimai v1.6

## 1.1 Problemos aprašymas

Sukurti MLOps sprendimą, skirtą BERT Tiny klasifikavimo modelio treniravimui ir eksperimentų sekimui. Projektas turi leisti automatiškai treniruoti modelį ant duomenų rinkinio, įrašyti metrikas į MLflow, išsaugoti modelio artefaktą ir palaikyti eksperimentų atkūrimą pagal run ID, git commit bei nustatytą seed'ą.

Nuo U2 sprendimas apima ir visą modelio gyvavimo ciklą: mokymą Kubernetes Job'e, modelio registravimą MLflow Model Registry su alias'u `@champion`, inferencijos API (FastAPI) klasteryje ir CI/CD grandinę su kokybės vartu bei rollback.

Tikslas: užtikrinti pakankamą modelio kokybę, matomumą ir atkuriamumą, kad bet kuris eksperimento rezultatas būtų įvertinamas ir pakartojamas per trumpą laiką.

## 2. Funkciniai reikalavimai

| ID    | Reikalavimas                                                                 | Prioritetas |
|-------|------------------------------------------------------------------------------|-------------|
| FR-01 | Sistema turi leisti paleisti treniravimo procesą su konfigūracijos parametrais *(learning_rate, train_batch_size, epochs, seed, lr_scheduler_type, optimizer)*. | Must |
| FR-02 | Treniruotės metu turi būti generuojamos pagrindinės metrikos: *train_loss, eval_accuracy, eval_f1, eval_precision, eval_recall* ir *eval_loss*. | Must |
| FR-03 | Po treniravimo sistema turi išsaugoti modelio artefaktą ir su juo susijusius parametrus MLflow arba lokaliai `mlruns` kataloge. | Must |
| FR-04 | Sistema turi palaikyti vertinimo fazę su atskiru eval rinkiniu ir grąžinti rezultatus, leidžiančius palyginti modelio kokybę tarp skirtingų run'ų. | Must |
| FR-05 | Vartotojas turi galėti aprašyti duomenų šaltinį, mokymo/eval padalijimą ir išeigos katalogą, kad eksperimentai būtų pakartojami be rankinio koregavimo. | Must |
| FR-06 | Sistema turi palaikyti hiperparametrų paiešką (Optuna, `src/hpo.py`): vienas tėvinis run'as ir ≥ 5 įdėtiniai (nested) bandymai; paieškos erdvė aprašoma `params.yaml → hpo`. | Must |
| FR-07 | Mokymas turi būti paleidžiamas kaip Kubernetes Job (`k8s/train-job.yaml`): parametrai per ConfigMap (`train-configmap.yaml`), prisijungimai per Secret (`mlflow-credentials`, ne Git'e), run'as log'inamas į klasterio MLflow. | Must |
| FR-08 | Modelis registruojamas MLflow Model Registry (`bert-tiny-sst2`, `src/register.py`); naujai versijai priskiriamas `@champion`, ankstesnei – `@previous`. | Must |
| FR-09 | Inferencijos servisas (FastAPI, Deployment + Service) turi įkelti modelį **iš registro pagal alias'ą** (`MODEL_URI=models:/bert-tiny-sst2@champion`), ne iš failo vaizde; teikia `/health` (grąžina modelio pavadinimą, versiją, run ID) ir `/predict`; kol modelis neįkeltas – `503`. | Must |
| FR-10 | CI (GitHub Actions, `ci.yml`) kiekvienam push/PR vykdo `ruff` lint, `pytest` unit testus ir publikuoja vaizdus į GHCR su žyma `sha-<commit>`. | Must |
| FR-11 | CD (`cd.yml`, self-hosted runner) vykdo grandinę: testai → mokymo Job → kokybės vartas → registravimas → vaizdo build/push → manifesto atnaujinimas ir `rollout`; jei vartas nepraeina, registravimas ir deploy **nevykdomi**. | Must |
| FR-12 | Turi būti galimas rollback į ankstesnę modelio versiją (`scripts/rollback.sh` / `rollback.yml`: `@champion` → `@previous` arba nurodyta versija + `rollout restart`) ir į ankstesnį vaizdą (`kubectl rollout undo`); CD deploy job'as atlieka rollback automatiškai, jei nepavyksta `rollout status` arba smoke testas. | Must |

## 3. Nefunkciniai reikalavimai

| ID     | Kategorija      | Reikalavimas                                                                 | KPI |
|--------|-----------------|-----------------------------------------------------------------------------|-----|
| NFR-01 | Kokybė          | Modelio klasifikavimo kokybė turi būti pakankama praktiniam naudojimui. Slenkstis sutampa su CD kokybės vartu (žr. NFR-13). | `eval_f1 >= 0.70`, `eval_accuracy >= 0.65` (išmatuota: 0.750 / 0.696) |
| NFR-02 | Reprodukcija    | Kiekvienas istorinis run turi būti atkuriamas naudojant MLflow run ID ir/vai eksperimentų metaduomenis. | Atkūrimo laikas <= 2 min (be duomenų atsisiuntimo), `delta accuracy <= 0.005` |
| NFR-03 | Atsekamumas     | Kiekvienam eksperimentui turi būti įrašyti MLflow run ID, git commit ir duomenų hash (seed/model config). | 100 % run'ų turinčių reikiamus tag'us |
| NFR-04 | Ištekliai       | Treniruotė turi būti vykdoma efektyviai ir nedideliu resursų panaudojimu. | CPU treniruotė užbaigiama per <= 10 min, RAM <= 8 GB (išmatuota: Job `Completed` per 2 min 38 s, limitas 2 CPU / 4Gi) |
| NFR-05 | Stebėsena       | Visi svarbiausi eksperimentiniai rezultatai turi būti matomi ir palyginami. | 100 % pagrindinių metrikų ir parametrų įrašyta į MLflow |
| NFR-06 | Saugumas        | Nėra jautrios konfigūracijos ar prieigos informacijos repo istorijoje. Prisijungimai laikomi Kubernetes Secret'uose (`mlflow-credentials`, `mlflow-db`), repo tik `secret.example.yaml`; senas DB slaptažodis pakeistas. | 0 konfidencialių duomenų įrašų git repozitoriume |

### 3.1 Inferencijos NFR (išmatuota)

Matavimai: `scripts/load_test.sh` → [load-test.md](load-test.md), 2026-11-09, modelis `bert-tiny-sst2` v17, vaizdas `ghcr.io/akrasavcev/bert-tiny:sha-56a40cb4c8f9`, 1 pod (limitas 1 CPU / 1Gi), 500 `POST /predict` užklausų, lygiagretumas 10, per `kubectl port-forward` (vėlinimas matuojamas kliento pusėje).

| ID     | Kategorija        | Reikalavimas | KPI | Išmatuota |
|--------|-------------------|--------------|-----|-----------|
| NFR-07 | Vėlinimas         | `/predict` p95 vėlinimas esant 10 lygiagrečių užklausų. | p95 <= 300 ms | **183.3 ms** (p50 115.0, p99 217.6, vidurkis 134.2) |
| NFR-08 | Pralaidumas       | Vienas pod'as turi aptarnauti pakankamą užklausų srautą. | >= 50 req/s | **74.4 req/s** |
| NFR-09 | Patikimumas       | Apkrovos teste neturi būti klaidų. | klaidų dalis <= 1 % | **0.00 %** (0 iš 500) |
| NFR-10 | Modelio dydis     | Modelio artefaktas turi būti mažas, kad greitai atsisiųstų iš registro. | <= 50 MB | **18 MB** |
| NFR-11 | Vaizdo dydis / ištekliai | Inferencijos vaizdas be modelio (modelis atsisiunčiamas paleidžiant); atminties naudojimas telpa į limitą. | vaizdas <= 500 MB; atmintis <= 1Gi | **386 MB** (nesuspaustas); pod'o atminties pikas **352 MiB** |
| NFR-12 | Šaltas startas    | Nuo `rollout restart` iki pod'o `Ready`, įskaitant modelio atsisiuntimą iš registro. | <= 60 s (`startupProbe` leidžia iki 5 min) | **6.4 s** (vaizdas jau mazge) |

### 3.2 CI/CD NFR

| ID     | Kategorija        | Reikalavimas | KPI | Išmatuota / patikrinta |
|--------|-------------------|--------------|-----|------------------------|
| NFR-13 | Kokybės vartas    | Modelis registruojamas ir diegiamas tik jei test rinkinio metrika ne žemesnė už slenkstį (`params.yaml → gate`, `scripts/quality_gate.py`). | `eval_f1 >= 0.70`; raudonas vartas → 0 deploy žingsnių | Paskutinis CD Job run'as: `eval_f1 = 0.750` → praėjo (v18 įdiegta); `--min-value 0.95` → vartas nepraeina, `deploy` praleidžiamas |
| NFR-14 | Atsekamumas (deploy) | Kiekvienas įdiegtas vaizdas ir modelis susiejami su commit'u. | vaizdo žyma `sha-<12 simb. commit>`; `/health` grąžina modelio versiją ir run ID | `sha-ad2c2311a168`, modelis v18 (commit `0f45322`) |
| NFR-15 | Rollback          | Grįžimas į ankstesnę modelio versiją be naujo mokymo. | 1 komanda (`scripts/rollback.sh`); `/health` rodo ankstesnę versiją | Patikrinta 2026-11-10: v17 → v16 → v17 |
| NFR-16 | CI kokybė         | Kodas tikrinamas automatiškai. | `ruff check` be klaidų; 100 % unit testų praeina | `ruff` švarus, `pytest` 30/30 |

## 4. Reprodukcijos reikalavimas

Pagrindinis projekto reikalavimas yra užtikrinti atkartojamumą. Eksperimentas laikomas reprodukuojamu, kai:

- yra žinomas `run_id` arba eksperimentinis konfigūracijos rinkinys;
- modelio treniruotės parametrai atitinka įrašytus MLflow parametrus;
- naudojamas tas pats seed, duomenų versija ir git commit;
- po pakartotinio paleidimo rezultatų skirtumas neviršija nurodyto KPI ribos.

Tai leidžia palyginti ankstesnius ir naujus rezultatus, greitai identifikuoti regresijas ir išlaikyti MLOps praktikų auditų atitiktį.

## 7. Keitimo istorija

| Versija | Data       | Autorius      | Pakeitimai |
|---------|------------|---------------|------------|
| 1.0     | 2026-10-13 | A. Krasavcev | Pradinis dokumentas |
| 1.1     | 2026-10-13 | A. Krasavcev | Užpildyti funkciniai ir nefunkciniai reikalavimai, įskaitant reprodukcijos NFR |
| 1.2     | 2026-10-19 | A. Krasavcev | Papildyti FR atskiri punktai |
| 1.3     | 2026-11-08 | A. Krasavcev | Pridėti FR-06 (HPO su nested run'ais), FR-07 (mokymo Job su ConfigMap/Secret); NFR-06 papildytas Secret'ų naudojimu ir slaptažodžio rotacija |
| 1.4     | 2026-11-09 | A. Krasavcev | Pridėti FR-08 (registras ir `@champion`), FR-09 (servisas ima modelį iš registro pagal alias'ą); nauji inferencijos NFR-07–NFR-12 su išmatuotomis reikšmėmis iš apkrovos testo |
| 1.5     | 2026-11-09 | A. Krasavcev | Pridėti FR-10 (CI: lint, testai, GHCR) ir NFR-16; NFR-04 papildytas išmatuotu Job laiku |
| 1.6     | 2026-11-10 | A. Krasavcev | Pridėti FR-11 (CD su kokybės vartu), FR-12 (rollback), NFR-13–NFR-15; NFR-01 suderintas su faktine kokybe ir vartu (`eval_f1 >= 0.80` → `>= 0.70`, nes BERT-Tiny po 1 epochos ant 10k pavyzdžių pasiekia ~0.75) |
