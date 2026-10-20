# End-to-End Heart Disease MLOps Report

> Complete this report after running the project. Replace every bracketed field with observed results and add original analysis. The current workspace does not include deployment credentials, screenshots, a video, or a GitHub remote, so these items are intentionally left for the student to produce and verify.

## 1. Executive summary

**Objective:** Build and operationalize a reproducible binary classifier using the UCI Heart Disease Cleveland dataset.

**Selected model:** Logistic Regression (`C=0.1`), selected by mean stratified five-fold training ROC-AUC.

**Holdout metrics:** Accuracy 0.885, precision 0.839, recall 0.929, ROC-AUC 0.965. The positive class is disease present. In a medical screening context, missed positive cases are a serious concern, but these metrics are from a small historical dataset and do not justify clinical use.

**Deployment status:** [Local Docker / local Kubernetes / public cloud, with verified URL or access instructions]

## 2. Data and acquisition

The project downloads the official UCI Heart Disease archive (dataset DOI: 10.24432/C52P4X), extracts the processed Cleveland data, and maps the original angiographic `num` target to absence (`0`) versus presence (`1`-`4`). The Cleveland subset contains 303 observations and 13 predictors. The raw file represents unknowns with `?`; the loader converts these to missing values. Record the actual class counts and missingness from your run here: [ ].

## 3. Exploratory analysis

Generated figures: `artifacts/eda/distributions.png` and `artifacts/eda/correlation.png`. Describe the observed class balance, age distribution, notable correlations, and any limitations of interpreting these patterns from a small dataset: [ ]. Insert original figure screenshots or embed the generated images in the submitted report.

## 4. Feature processing and model selection

The inference and training schema contains age, sex, chest-pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, ST depression, ST slope, fluoroscopy vessel count, and thalassemia category. Numeric fields receive median imputation and standard scaling. Categorical fields receive most-frequent imputation and one-hot encoding. All transformations reside in the sklearn pipeline and are fitted within cross-validation folds.

Candidates: Logistic Regression and Random Forest. Selection uses a stratified five-fold CV ROC-AUC grid search on the training split, with a stratified 80/20 holdout used for final reporting. Hyperparameter grids are defined in `scripts/train.py`. Fill this comparison from the MLflow experiment and `artifacts/training_summary.json`:

| Model | CV ROC-AUC | Holdout accuracy | Precision | Recall | Holdout ROC-AUC | Best parameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | 0.903 | 0.885 | 0.839 | 0.929 | 0.965 | `C=0.1` |
| Random Forest | 0.902 | 0.869 | 0.813 | 0.929 | 0.959 | `max_depth=None`, `min_samples_leaf=3` |

The two candidates have very similar CV ROC-AUC, with Logistic Regression ahead by about 0.001. On the fixed holdout, both have the same recall, while Logistic Regression has slightly higher accuracy, precision, and ROC-AUC. It is therefore selected as a compact baseline for this experiment, not as evidence of clinical suitability. Explain any further interpretation after reviewing the generated confusion matrix and ROC plot: [ ].

## 5. Experiment tracking and reproducibility

MLflow experiment: `uci-heart-disease-cleveland`, stored locally in `mlflow.db`. Each candidate run records model identity, CV selection parameters, seed, fold count, metrics, a classification report, evaluation plot, and the sklearn pipeline. The selected packaged model and summary are saved beneath `artifacts/`. Record the MLflow run IDs and dashboard screenshot: [ ].

## 6. Testing and CI/CD

Pytest covers raw parsing, unknown values, target mapping, missing feature validation, and end-to-end pipeline fitting/prediction. GitHub Actions runs Ruff, pytest, model training, and EDA, then uploads artifacts. Add the repository URL and a screenshot/link to the passing workflow run: [ ].

## 7. Serving, container and deployment

FastAPI exposes `/health`, `/predict`, and `/metrics`. The prediction response includes the binary label and class confidence; request logs omit patient feature values. The Docker image contains the pre-trained serialized pipeline. Kubernetes manifests provide two replicas, readiness/liveness probes, resource limits, and a LoadBalancer Service.

**Docker build and sample request evidence:** Not executed in the current environment; capture after Docker is available: [ ]

**Deployment target and verified endpoint/access instructions:** Kubernetes manifests are prepared, but no cluster or cloud endpoint is available in the current environment: [ ]

**Kubernetes rollout evidence or cloud deployment URL:** Not yet produced; attach genuine verification after deployment: [ ]

## 8. Monitoring and operational considerations

The Prometheus client exports request totals by method/path/status and request-duration histograms. Configure Prometheus using `deployment/prometheus.yml`; connect Grafana to Prometheus and create dashboard panels for request rate, error rate, and latency. Attach genuine screenshots and state the scrape target/status: [ ]. Do not log request bodies or retain patient data in application logs.

## 9. Limitations, risks and next steps

Discuss small sample size, missing data, historical/cohort bias, distribution shift, lack of external validation, classification threshold trade-offs, and the distinction between educational prediction and clinical decision support: [ ]. Suggested next steps include external validation, calibration, subgroup performance analysis, model/data versioning, secrets and image scanning, alerting, and controlled rollout.

## 10. Evidence and references

- UCI Machine Learning Repository, Heart Disease dataset, DOI: <https://doi.org/10.24432/C52P4X>.
- Code repository: [URL]
- MLflow experiment screenshot/run IDs: [ ]
- CI workflow screenshot/link: [ ]
- Docker/API test screenshots: [ ]
- Kubernetes/cloud deployment screenshots and access instructions: [ ]
- Short pipeline walkthrough video: [location/link]# End-to-End Heart Disease MLOps Report

> Complete the remaining bracketed fields with your own interpretation and submission evidence. The project has been trained and its Docker API smoke-tested locally; this workspace still has no configured GitHub remote or Kubernetes context, and genuine screenshots/video must be captured by the student.

## 1. Executive summary

**Objective:** Build and operationalize a reproducible binary classifier using the UCI Heart Disease Cleveland dataset.

**Selected model:** [from `artifacts/training_summary.json`]

**Holdout metrics:** Accuracy [ ], precision [ ], recall [ ], ROC-AUC [ ]. Record the positive class as disease present and explain which error type is most consequential in this educational context.

**Deployment status:** Local Docker image build and API smoke test passed. Kubernetes/cloud deployment is not yet verified.

## 2. Data and acquisition

The project downloads the official UCI Heart Disease archive (dataset DOI: 10.24432/C52P4X), extracts the processed Cleveland data, and maps the original angiographic `num` target to absence (`0`) versus presence (`1`-`4`). The Cleveland subset contains 303 observations and 13 predictors. The raw file represents unknowns with `?`; the loader converts these to missing values. Record the actual class counts and missingness from your run here: [ ].

## 3. Exploratory analysis

Generated figures: `artifacts/eda/distributions.png` and `artifacts/eda/correlation.png`. Describe the observed class balance, age distribution, notable correlations, and any limitations of interpreting these patterns from a small dataset: [ ]. Insert original figure screenshots or embed the generated images in the submitted report.

## 4. Feature processing and model selection

The inference and training schema contains age, sex, chest-pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, ST depression, ST slope, fluoroscopy vessel count, and thalassemia category. Numeric fields receive median imputation and standard scaling. Categorical fields receive most-frequent imputation and one-hot encoding. All transformations reside in the sklearn pipeline and are fitted within cross-validation folds.

Candidates: Logistic Regression and Random Forest. Selection uses a stratified five-fold CV ROC-AUC grid search on the training split, with a stratified 80/20 holdout used for final reporting. Hyperparameter grids are defined in `scripts/train.py`. Fill this comparison from the MLflow experiment and `artifacts/training_summary.json`:

| Model | CV ROC-AUC | Holdout accuracy | Precision | Recall | Holdout ROC-AUC | Best parameters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Random Forest | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

Explain why the selected operating point and error profile are appropriate for this assignment, and why this model must not be used for diagnosis: [ ].

## 5. Experiment tracking and reproducibility

MLflow experiment: `uci-heart-disease-cleveland`. Each candidate run records model identity, CV selection parameters, seed, fold count, metrics, a classification report, evaluation plot, and the sklearn pipeline. The selected packaged model and summary are saved beneath `artifacts/`. Record the MLflow run IDs and dashboard screenshot: [ ].

## 6. Testing and CI/CD

Pytest covers raw parsing, unknown values, target mapping, missing feature validation, and end-to-end pipeline fitting/prediction. GitHub Actions runs Ruff, pytest, model training, and EDA, then uploads artifacts. Add the repository URL and a screenshot/link to the passing workflow run: [ ].

## 7. Serving, container and deployment

FastAPI exposes `/health`, `/predict`, and `/metrics`. The prediction response includes the binary label and class confidence; request logs omit patient feature values. The Docker image contains the pre-trained serialized pipeline. Kubernetes manifests provide two replicas, readiness/liveness probes, resource limits, and a LoadBalancer Service.

**Docker build and sample request evidence:** [ ]

**Deployment target and verified endpoint/access instructions:** [ ]

**Kubernetes rollout evidence or cloud deployment URL:** [ ]

## 8. Monitoring and operational considerations

The Prometheus client exports request totals by method/path/status and request-duration histograms. Configure Prometheus using `deployment/prometheus.yml`; connect Grafana to Prometheus and create dashboard panels for request rate, error rate, and latency. Attach genuine screenshots and state the scrape target/status: [ ]. Do not log request bodies or retain patient data in application logs.

## 9. Limitations, risks and next steps

Discuss small sample size, missing data, historical/cohort bias, distribution shift, lack of external validation, classification threshold trade-offs, and the distinction between educational prediction and clinical decision support: [ ]. Suggested next steps include external validation, calibration, subgroup performance analysis, model/data versioning, secrets and image scanning, alerting, and controlled rollout.

## 10. Evidence and references

- UCI Machine Learning Repository, Heart Disease dataset, DOI: <https://doi.org/10.24432/C52P4X>.
- Code repository: [URL]
- MLflow experiment screenshot/run IDs: [ ]
- CI workflow screenshot/link: [ ]
- Docker/API test screenshots: [ ]
- Kubernetes/cloud deployment screenshots and access instructions: [ ]
- Short pipeline walkthrough video: [location/link]