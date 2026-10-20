# Heart Disease MLOps

**Repository:** [arunkarthick2003/mlops-assignment1-a3095887](https://github.com/arunkarthick2003/mlops-assignment1-a3095887)
**Course:** Machine Learning Operations (MLOps), Assignment 01

An end-to-end, reproducible machine-learning project using the Cleveland subset of the UCI Heart Disease dataset. It covers data acquisition, exploratory analysis, model comparison, MLflow experiment tracking, automated testing, a FastAPI service, Docker packaging, local Kubernetes deployment, and request metrics/logging.

> Educational assignment only. This model is not a medical device and must not be used for diagnosis, treatment, or clinical decision-making.

## Results

The source target `num` is mapped to a binary label: `0` means no disease in the dataset, while original values `1`-`4` map to disease present. The dataset contains 303 records, with 164 negative and 139 positive labels. Six predictor values are missing (`ca`: 4, `thal`: 2); the preprocessing pipeline imputes them during training.

Stratified 5-fold CV was used for model selection on the training split. A separate stratified 80/20 holdout (61 rows) was used for final metrics.

| Candidate | CV ROC-AUC | Holdout accuracy | Precision | Recall | Holdout ROC-AUC | Selected parameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | 0.9028 | 0.8852 | 0.8387 | 0.9286 | 0.9654 | `C=0.1` |
| Random Forest | 0.9019 | 0.8689 | 0.8125 | 0.9286 | 0.9589 | `max_depth=None`, `min_samples_leaf=3` |

Logistic Regression was selected by mean CV ROC-AUC. On the holdout, the confusion matrix in label order `[absent, present]` is `[[28, 5], [2, 26]]`. These estimates are uncertain because the dataset and holdout are small and historical.

## Setup and training

Use Python 3.11 or 3.12. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts\download_data.py
python scripts\eda.py
python scripts\train.py
pytest -q
ruff check src scripts tests
```

The downloader retrieves the official UCI archive and writes the cleaned table to `data/processed/heart_disease.csv`. EDA images, evaluation plots, the serialized preprocessing/model pipeline, and a training summary are written under `artifacts/`. Generated raw data, model binaries, MLflow database files, and the virtual environment are excluded from Git; scripts recreate the model and experiment data.

### Experiment tracking

The experiment is named `uci-heart-disease-cleveland`. Start the local UI with:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000`, select **Model training**, then the experiment’s **Runs** view. Candidate runs log parameters, metrics, reports, and evaluation plots; the selected model and summary are logged as package artifacts. Candidate run IDs in the latest training summary are `83dcf44e9edf4296ac210cd0c43fcbbe` (Logistic Regression) and `5ba36aeea1d64c03a54d813b8a839b89` (Random Forest).

## API and Docker

Train first so the model file exists, then build and run:

```powershell
docker build -t heart-risk-api:local .
docker run --rm -p 8000:8000 heart-risk-api:local
```

Endpoints:

- `GET /health` reports whether the model loaded.
- `POST /predict` accepts the 13 numeric feature fields and returns a binary prediction, confidence for the predicted class, and an educational-use disclaimer.
- `GET /metrics` exposes Prometheus-format request counters and latency histograms.

Example JSON for `POST /predict`:

```json
{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":2,"ca":0,"thal":6}
```

Open `http://localhost:8000/docs` for interactive API testing. Logs include request ID, route, status, and latency; request bodies are not logged.

## Local Kubernetes

The verified target was local Kubernetes through Docker Desktop. Enable Kubernetes in Docker Desktop settings, then confirm the context:

```powershell
kubectl config current-context
```

It should report `docker-desktop`. The built image is already in Docker Desktop’s image store; do not run `minikube image load` for this path. Apply and verify:

```powershell
docker build -t heart-risk-api:local .
kubectl apply -f deployment/kubernetes.yaml
kubectl rollout status deployment/heart-risk-api
kubectl get deployment,pods,service
kubectl port-forward service/heart-risk-api 8080:80
```

While the port-forward command remains running, test `http://localhost:8080/health` and `http://localhost:8080/docs`. The verified rollout had two ready replicas. A LoadBalancer address shown by a local cluster is not a public endpoint; use port-forwarding for local access.

The alternative Minikube path requires installing Minikube and starting a cluster before running `minikube image load heart-risk-api:local`. Prometheus scrape configuration is provided in `deployment/prometheus.yml`. The API’s `/metrics` endpoint and request logs were verified; a Grafana dashboard is not included.

## CI and verification

GitHub Actions runs Ruff, training, pytest, EDA generation, and uploads artifacts. The latest recorded run for commit `1dcd8b9` completed successfully in 1m 27s: [View CI run](https://github.com/arunkarthick2003/mlops-assignment1-a3095887/actions/runs/39411595260).

Local verification completed: six pytest tests passed, Ruff passed, the Docker image built, API health/prediction/metrics responded, and Kubernetes reported 2/2 replicas ready. These checks do not constitute public cloud deployment.

## Project files and reports

- [Detailed report (Markdown)](REPORT.md)
- [Final report (DOCX)](Heart_Disease_MLOps_Report.docx)
- `scripts/`: data download, EDA, model training, and DOCX report generation.
- `src/heart_mlops/`: data, model, and API implementation.
- `tests/`: data/pipeline and API tests.
- `deployment/`: Kubernetes and Prometheus configuration.
- `screenshots/`: EDA figures and collected local CI, Docker, API, and Kubernetes evidence.

Regenerate the Word report after installing requirements with:

```powershell
python scripts\build_report.py
```

## Limitations

The Cleveland dataset is small, historical, and not representative of present-day patient populations. No independent external validation, calibration, prospective evaluation, or subgroup fairness assessment was performed. The 61-row holdout has high statistical uncertainty. Correlations and scores are descriptive model-development results, not evidence of clinical utility. The API is an educational demonstration and must not be used to make health decisions.
