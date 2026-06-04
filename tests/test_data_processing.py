import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing import process_transactions, pipeline


def test_process_transactions_adds_target():
    df = pd.read_csv("data/raw/data.csv")
    out = process_transactions(df)
    assert "is_high_risk" in out.columns
    # values should be 0 or 1
    vals = set(out["is_high_risk"].dropna().unique().tolist())
    assert vals.issubset({0, 1})


def test_pipeline_transforms():
    df = pd.read_csv("data/raw/data.csv")
    transformed = pipeline.fit_transform(df)
    # should return a pandas DataFrame or numpy array with at least one row
    assert transformed is not None
    if hasattr(transformed, "shape"):
        assert transformed.shape[0] > 0
