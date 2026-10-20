"""Download and validate the processed Cleveland Heart Disease dataset."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATASET_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET = "target"
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_FEATURES = [name for name in FEATURES if name not in CATEGORICAL_FEATURES]
COLUMNS = FEATURES + ["num"]


def load_cleveland_csv(path: str | Path) -> pd.DataFrame:
    """Read UCI's headerless processed Cleveland file and binarize its target."""
    frame = pd.read_csv(path, header=None, names=COLUMNS, na_values="?", skipinitialspace=True)
    if frame.shape[1] != len(COLUMNS):
        raise ValueError(f"Expected {len(COLUMNS)} columns, found {frame.shape[1]}.")
    frame[TARGET] = (pd.to_numeric(frame.pop("num"), errors="coerce") > 0).astype("int64")
    for column in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[TARGET].isna().any():
        raise ValueError("The target contains missing or invalid values.")
    return frame


def download_dataset(destination: str | Path = "data/raw/processed.cleveland.data") -> Path:
    """Download the official UCI archive and extract its Cleveland processed file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "heart-risk-mlops/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        member = next(
            (name for name in archive.namelist() if name.lower().endswith("processed.cleveland.data")),
            None,
        )
        if member is None:
            raise FileNotFoundError("UCI archive did not contain processed.cleveland.data.")
        destination.write_bytes(archive.read(member))
    return destination


def load_or_download(path: str | Path = "data/raw/processed.cleveland.data") -> pd.DataFrame:
    """Load a local copy, downloading it from UCI when needed."""
    path = Path(path)
    if not path.exists():
        download_dataset(path)
    return load_cleveland_csv(path)