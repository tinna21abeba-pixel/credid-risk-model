import pandas as pd
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.data_processing import pipeline

df= pd.read_csv(
    "data/raw/data.csv"
)
x = pipeline.fit_transform(df)
print(x.shape)