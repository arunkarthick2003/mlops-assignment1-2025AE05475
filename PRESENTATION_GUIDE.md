# Presentation Walkthrough Guide

Use this run-of-show while recording the project presentation. It follows the speech from introduction through limitations and identifies what to open or show on screen. Suggested duration: 4-5 minutes.

## Before Recording

1. Open the GitHub repository: <https://github.com/arunkarthick2003/mlops-assignment1-a3095887>.
2. Keep the following local applications/windows ready: VS Code at the project root, browser at the local MLflow UI, browser at the API docs, and PowerShell terminals for Kubernetes/API commands.
3. Confirm the local Kubernetes deployment is ready:

   ```powershell
   kubectl get deployment,pods,service
   kubectl rollout status deployment/heart-risk-api
   ```

   Expected: Deployment shows 2/2 available replicas, pods are Running/Ready, and rollout says successfully rolled out.
4. If the MLflow UI is not already running, open a terminal at the project root and start it. Leave this terminal running:

   ```powershell
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```

   Open `http://127.0.0.1:5000`. In MLflow, select **Model training**, open `uci-heart-disease-cleveland`, then choose **Runs**.
5. If the Kubernetes API port-forward is not already running, open another terminal and leave it running:

   ```powershell
   kubectl port-forward service/heart-risk-api 8080:80
   ```

   Open `http://localhost:8080/docs`. Use `http://localhost:8080/health` to show the health response.
6. Close unrelated windows and notifications. Do not show access tokens, passwords, or patient information. This is a public educational dataset and demonstration; do not use real patient data.

## Recording Sequence

### 1. Introduction | 0:00-0:30

**Open:** GitHub repository front page, then `README.md` in VS Code.

**Show:** Repository name and the README overview/architecture.

**Say:** This is an end-to-end MLOps project for the UCI Cleveland heart-disease dataset. It includes data acquisition, model training, experiment tracking, testing, API serving, Docker, local Kubernetes, and basic monitoring. It is educational only, not a medical diagnostic tool.

### 2. Data and EDA | 0:30-1:00

**Open:** `data/processed/heart_disease.csv`, then `screenshots/eda-distributions.png` and `screenshots/eda-correlation.png`.

**Show:** Dataset columns/rows briefly, then the distribution/class-balance figure and correlation heatmap.

**Say:** The data has 303 records, 13 predictors, 164 negative and 139 positive targets. Six predictor values are missing: four `ca` values and two `thal` values. The pipeline handles these with fitted imputers. Correlations are descriptive, not causal.

### 3. Models and Results | 1:00-1:45

**Open:** `artifacts/training_summary.json` and `screenshots/model-results.png`.

**Show:** The selected model and the table comparing CV and holdout metrics, confusion matrices, and ROC curves.

**Say:** Logistic Regression with `C=0.1` was selected by stratified five-fold CV ROC-AUC. Its holdout accuracy was 0.885, precision 0.839, recall 0.929, and ROC-AUC 0.965. The holdout has only 61 rows, so these numbers are uncertain and are not evidence of clinical utility.

### 4. Experiment Tracking | 1:45-2:15

**Open:** The running MLflow page at `http://127.0.0.1:5000`.

**Show:** The `uci-heart-disease-cleveland` experiment in **Model training > Runs**. Point to completed Logistic Regression and Random Forest runs and their metrics.

**Say:** MLflow records parameters, metrics, reports, and evaluation artifacts for both candidates. The selected model and summary are saved as artifacts. The image `screenshots/mlflow-runs.png` is a formatted metrics export, not a raw MLflow browser screenshot.

### 5. Tests and CI | 2:15-2:45

**Open:** `tests/test_pipeline.py`, `tests/test_api.py`, then `screenshots/github-actions.png` or the GitHub Actions run page.

**Show:** A few test names, then the successful CI run for commit `1dcd8b9`.

**Say:** The six tests cover data parsing, target conversion, preprocessing/model predictions, and API behavior. Ruff and pytest passed locally. GitHub Actions also completed successfully; it runs linting, training, tests, and EDA.

### 6. Docker and API | 2:45-3:30

**Open:** `Dockerfile`, `src/heart_mlops/api.py`, and `screenshots/docker-build.png`. Then show `http://localhost:8080/docs` if the port-forward is running.

**Show:** The successful image build, the `/health` response, and a sample request to `POST /predict` with the returned label/confidence.

**Say:** The trained preprocessing/model pipeline is included in the Docker image and served by FastAPI. The service exposes health, prediction, and Prometheus metrics routes. Logs include request metadata and latency but do not log feature payloads.

### 7. Kubernetes and Monitoring | 3:30-4:15

**Open:** `deployment/kubernetes.yaml`, `screenshots/kubernetes-rollout.png`, `screenshots/api-metrics.png`, and `screenshots/api-logs.png`.

**Show:** The two-replica Deployment and Service, the successful rollout screenshot, and metrics/log evidence. If using a terminal, run `kubectl get deployment,pods,service` and `kubectl logs deployment/heart-risk-api --tail=20`.

**Say:** The API ran in local Docker Desktop Kubernetes with two ready replicas. I verified it via port-forwarding. The metrics endpoint and request logs demonstrate basic monitoring; this was not a public cloud deployment, and no Grafana dashboard was deployed.

### 8. Conclusion | 4:15-4:45

**Open:** `Heart_Disease_MLOps_Report.docx` or `REPORT.md`.

**Say:** The project demonstrates a reproducible path from UCI data to tracked experiments, CI, a containerized API, and local Kubernetes. The small historical dataset, lack of external validation, and lack of calibration mean the model must not be used for patient-care decisions. Mention any follow-up work honestly, including external validation, subgroup analysis, and the required walkthrough video if it has not yet been recorded.

## Final Checks

- Keep MLflow and port-forward terminals running while showing their pages.
- Do not claim a public endpoint; the verified deployment is local Kubernetes.
- Do not claim a Grafana dashboard; the delivered monitoring is API metrics and logs.
- Avoid lingering on duplicate/troubleshooting MLflow runs; use the two candidate runs and the clean comparison image.
- Add the final video filename/link to `REPORT.md` and push the finished report, README, DOCX, guide, and evidence to GitHub.
