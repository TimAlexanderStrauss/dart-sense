# Model Training Guide (Dart Sense)

## Kurzantwort zu Python 3.8

Ja, 3.8 ist für neue Modelle problematisch:

- Python 3.8 ist EOL (keine Sicherheitsupdates).
- Neue `torch`/`ultralytics` Releases droppen alte Python-Versionen zuerst.
- Mehr Risiko für Dependency-Konflikte und fehlende Wheels.

Empfehlung: **Projekt auf Python 3.12** umstellen und Training auf dem Windows-Desktop mit GPU ausführen.

## Hinweise zur Datenlage aus dem Projekt

- In `README.md` steht: Im Repo liegt nur ein **kleines Sample** im `data`-Ordner.
- Ebenfalls in der README: Das Trainingsskript muss für die eigene Maschine angepasst werden.
- Aktuell im Repo vorhanden:
  - `data/darts/images/small_sample`
  - `data/darts/labels/small_sample`

Für echtes Retraining brauchst du den vollständigen Datensatz lokal.

## Plan zur Umstellung auf Python 3.12

### Phase 1: Baseline einfrieren

1. Aktuellen Stand committen.
2. Dokumentieren, wie App und Training aktuell gestartet werden.
3. Referenzwerte (mAP/Precision/Recall, FPS) sichern.

### Phase 2: Umgebung modernisieren

1. Lokale `.venv` auf 3.12 nutzen.
2. Conda-Export-`requirements.txt` nicht mehr als primäre Installationsquelle verwenden.
3. Für Runtime/Training klare pip-basierte Installationsanweisungen verwenden.

### Phase 3: Plattformunabhängigkeit härten

1. Windows-Separatoren (`\\`) in Pfaden aus Altcode entfernen.
2. Für Pfade konsistent `pathlib`/`os.path` nutzen.
3. Modellpfad konfigurierbar halten (nicht hart verdrahtet).

### Phase 4: Trainingspfade trennen

1. Host/Container-Pfade nicht mischen.
2. Eigene Docker-YAML nutzen: `data/d1_to_d7.docker.yaml`.
3. Reproduzierbare Trainingskommandos in Skripten bereitstellen.

### Phase 5: Smoke-Tests

1. Imports + GUI-Start auf 3.12 prüfen.
2. Kurzen Inferenzlauf mit existierendem Modell durchführen.
3. Danach erstes Baseline-Retraining starten.

## Docker-Setup für Windows + NVIDIA GPU

Bereitgestellte Dateien:

- `docker/Dockerfile.train`
- `docker-compose.train.yml`
- `data/d1_to_d7.docker.yaml`
- `training/train_docker.ps1`
- `training/train_docker.sh`
- `training/train_optimal.sh` (historische Hyperparameter, jetzt Docker-basiert)

### Voraussetzungen auf deinem Windows-Desktop

1. Aktueller NVIDIA-Treiber.
2. Docker Desktop mit aktivierter WSL2-Engine.
3. GPU-Test:

```powershell
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Schnellstart Training (Windows PowerShell)

```powershell
.\training\train_docker.ps1
```

Mit Parametern:

```powershell
.\training\train_docker.ps1 -Model yolov8s.pt -RunName v8s_run1 -Batch 8 -Epochs 120
```

AutoBatch:

```powershell
.\training\train_docker.ps1 -Batch "-1" -RunName autobatch_test
```

Oder mit Compose:

```powershell
docker compose -f docker-compose.train.yml up --build
```

## RTX 5070 (12 GB VRAM): Batch-Empfehlungen

Startwerte bei `imgsz=800`:

- `yolov8n`: `batch=16` (typischer Bereich 12-24)
- `yolov8s`: `batch=8` (typischer Bereich 6-12)
- `yolov8m`: `batch=4` (typischer Bereich 3-6)

Pragmatisches Vorgehen:

1. Erst mit konservativem Batch starten.
2. Bei stabiler Auslastung schrittweise erhöhen.
3. Bei OOM sofort halbieren.

Hinweise:

- `batch=-1` aktiviert AutoBatch (nutzt verfügbare VRAM-Kapazität automatisch). Das ist im `training/train_optimal.sh` bereits so gesetzt.
- Größere `imgsz` kostet deutlich mehr VRAM als ein Modellwechsel.
- `workers` auf Windows oft bei `4-8` stabil; bei Problemen auf `0-2` reduzieren.

## Was du vor dem ersten echten Training anpassen musst

1. Vollen Datensatz in `data/darts` ablegen (nicht nur `small_sample`).
2. Prüfen, dass die Ordner aus YAML wirklich existieren:
   `images/d1...d7` und `labels/d1...d7` inkl. `train/val/test`.
3. Falls deine Struktur anders ist, `data/d1_to_d7.docker.yaml` anpassen.
4. Optional Run-Namen/Projektordner setzen, damit Vergleichsläufe sauber getrennt sind.

## Modell verbessern und neu trainieren

Empfohlene Reihenfolge:

1. Baseline-Lauf (`yolov8n`, 100 Epochen).
2. Vergleichslauf (`yolov8s`) mit ähnlichen Settings.
3. Beste Variante weiter tunen (Augmentation/LR/Batch).
4. Ergebnisse in der App validieren (nicht nur mAP, sondern reale Score-Qualität).

Bestes Modell liegt i. d. R. unter:

```text
runs/detect/<run_name>/weights/best.pt
```

Für die App:

1. nach `weights.pt` im Projektroot kopieren  
oder
2. Modellpfad explizit beim Start setzen.
