FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_PATH=/app/artifacts/heart_disease_model.joblib PYTHONPATH=/app/src
WORKDIR /app
COPY requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir --no-compile -r requirements.txt
COPY artifacts/heart_disease_model.joblib ./artifacts/heart_disease_model.joblib
USER 10001
EXPOSE 8000
CMD ["uvicorn", "heart_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]