import pandas as pd
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.data_processing import process_transactions
from src.data_processing import pipeline


def test_pipeline_object():
	from sklearn.pipeline import Pipeline as SKPipeline

	assert isinstance(pipeline, SKPipeline)


