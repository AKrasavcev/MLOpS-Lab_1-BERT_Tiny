# Lab 1 - Reikalavimai v1.2

## 1.1 Problemos aprašymas

Sukurti MLOps sprendimą, skirtą BERT Tiny klasifikavimo modelio treniravimui ir eksperimentų sekimui. Projektas turi leisti automatiškai treniruoti modelį ant duomenų rinkinio, įrašyti metrikas į MLflow, išsaugoti modelio artefaktą ir palaikyti eksperimentų atkūrimą pagal run ID, git commit bei nustatytą seed'ą.

Tikslas: užtikrinti pakankamą modelio kokybę, matomumą ir atkuriamumą, kad bet kuris eksperimento rezultatas būtų įvertinamas ir pakartojamas per trumpą laiką.

## 2. Funkciniai reikalavimai

| ID    | Reikalavimas                                                                 | Prioritetas |
|-------|------------------------------------------------------------------------------|-------------|
| FR-01 | Sistema turi leisti paleisti treniravimo procesą su konfigūracijos parametrais *(learning_rate, train_batch_size, epochs, seed, lr_scheduler_type, optimizer)*. | Must |
| FR-02 | Treniruotės metu turi būti generuojamos pagrindinės metrikos: *train_loss, eval_accuracy, eval_f1, eval_precision, eval_recall* ir *eval_loss*. | Must |
| FR-03 | Po treniravimo sistema turi išsaugoti modelio artefaktą ir su juo susijusius parametrus MLflow arba lokaliai `mlruns` kataloge. | Must |
| FR-04 | Sistema turi palaikyti vertinimo fazę su atskiru eval rinkiniu ir grąžinti rezultatus, leidžiančius palyginti modelio kokybę tarp skirtingų run'ų. | Must |
| FR-05 | Vartotojas turi galėti aprašyti duomenų šaltinį, mokymo/eval padalijimą ir išeigos katalogą, kad eksperimentai būtų pakartojami be rankinio koregavimo. | Must |

## 3. Nefunkciniai reikalavimai

| ID     | Kategorija      | Reikalavimas                                                                 | KPI |
|--------|-----------------|-----------------------------------------------------------------------------|-----|
| NFR-01 | Kokybė          | Modelio klasifikavimo kokybė turi būti pakankama praktiniam naudojimui. | `eval_f1 >= 0.80`, `eval_accuracy >= 0.80` |
| NFR-02 | Reprodukcija    | Kiekvienas istorinis run turi būti atkuriamas naudojant MLflow run ID ir/vai eksperimentų metaduomenis. | Atkūrimo laikas <= 2 min (be duomenų atsisiuntimo), `delta accuracy <= 0.005` |
| NFR-03 | Atsekamumas     | Kiekvienam eksperimentui turi būti įrašyti MLflow run ID, git commit ir duomenų hash (seed/model config). | 100 % run'ų turinčių reikiamus tag'us |
| NFR-04 | Ištekliai       | Treniruotė turi būti vykdoma efektyviai ir nedideliu resursų panaudojimu. | CPU treniruotė užbaigiama per <= 10 min, RAM <= 8 GB |
| NFR-05 | Stebėsena       | Visi svarbiausi eksperimentiniai rezultatai turi būti matomi ir palyginami. | 100 % pagrindinių metrikų ir parametrų įrašyta į MLflow |
| NFR-06 | Saugumas        | Nėra jautrios konfigūracijos ar prieigos informacijos repo istorijoje. | 0 konfidencialių duomenų įrašų viename git repozitoriume |

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