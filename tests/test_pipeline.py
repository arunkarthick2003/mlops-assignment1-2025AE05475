from __future__ import annotations

import pandas as pd
import pytest

from heart_mlops.data import FEATURES, load_cleveland_csv
from heart_mlops.model import build_model, validate_features


def test_loader_converts_unknowns_to_missing_and_target_to_binary(tmp_path):
    row = ["63", "1", "1", "145", "233", "1", "2", "150", "0", "2.3", "3", "0", "6", "2"]
    row[11] = "?"
    source = tmp_path / "sample.data"
    source.write_text(",".join(row), encoding="utf-8")

    frame = load_cleveland_csv(source)

    assert frame.loc[0, "target"] == 1
    assert pd.isna(frame.loc[0, "ca"])
    assert list(frame.columns) == FEATURES + ["target"]


def test_loader_maps_zero_target_to_absent(tmp_path):
    row = "63,1,1,145,233,1,2,150,0,2.3,3,0,6,0"
    source = tmp_path / "sample.data"
    source.write_text(row, encoding="utf-8")
    assert load_cleveland_csv(source).loc[0, "target"] == 0


def test_feature_validation_requires_all_fields():
    with pytest.raises(ValueError, match="Missing required feature"):
        validate_features(pd.DataFrame({"age": [60]}))


def test_pipeline_fits_with_missing_and_categorical_values():
    rows = []
    for index in range(30):
        row = {feature: float((index + offset) % 5) for offset, feature in enumerate(FEATURES)}
        row.update({"sex": index % 2, "cp": index % 4 + 1, "fbs": index % 2,
                    "restecg": index % 3, "exang": index % 2, "slope": index % 3 + 1,
                    "ca": index % 4, "thal": index % 5 + 3})
        rows.append(row)
    features = pd.DataFrame(rows, columns=FEATURES)
    features.loc[0, "chol"] = float("nan")
    model = build_model("logistic_regression")

    model.fit(features, pd.Series([index % 2 for index in range(len(features))]))

    assert model.predict(features.iloc[:2]).shape == (2,)
    assert model.predict_proba(features.iloc[:2]).shape == (2, 2)