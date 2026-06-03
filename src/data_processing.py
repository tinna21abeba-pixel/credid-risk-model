
"""Data processing and feature engineering utilities."""

from typing import Any
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xverse.transformer import WOE



def process_transactions(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("process_transactions is not implemented yet")


class aggregateFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # normalize common column name variants to the expected TitleCase names
        col_map = {
            "customerid": "CustomerId",
            "customer_id": "CustomerId",
            "amount": "Amount",
            "transactionstarttime": "TransactionStartTime",
            "transaction_start_time": "TransactionStartTime",
        }

        existing = {c.lower(): c for c in X.columns}
        for key, target in col_map.items():
            if key in existing and target not in X.columns:
                X = X.rename(columns={existing[key]: target})

        agg = X.groupby("CustomerId").agg(
            TotalTransactionAmount=("Amount", "sum"),
            AvgTransactionAmount=("Amount", "mean"),
            TransactionCount=("Amount", "count"),
            StdTransactionAmount=("Amount", "std")
        ).reset_index()

        X = X.merge(
            agg,
            on="CustomerId",
            how="left"
        )

        return X


class DateFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X["TransactionStartTime"] = pd.to_datetime(
            X["TransactionStartTime"]
        )

        X["TransactionHour"] = (
            X["TransactionStartTime"].dt.hour
        )

        X["TransactionDay"] = (
            X["TransactionStartTime"].dt.day
        )

        X["TransactionMonth"] = (
            X["TransactionStartTime"].dt.month
        )

        X["TransactionYear"] = (
            X["TransactionStartTime"].dt.year
        )

        return X


categorical_columns = [
    "ProviderId",
    "ProductCategory",
    "ChannelId",
    "CurrencyCode"
]

numerical_columns = [
    "Amount",
    "Value",
    "TransactionHour",
    "TransactionDay",
    "TransactionMonth",
    "TransactionYear",
    "TotalTransactionAmount",
    "AvgTransactionAmount",
    "TransactionCount",
    "StdTransactionAmount"
]


numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


preprocessor = ColumnTransformer([
    (
        "num",
        numerical_pipeline,
        numerical_columns
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_columns
    )
])
class WOETransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # If no labels provided, skip fitting WOE and act as passthrough
        if y is None:
            self.use_woe = False
            return self

        # attempt to fit WOE; if it fails, fall back to passthrough
        try:
            self.woe = WOE()
            self.woe.fit(X, y)
            self.use_woe = True
        except Exception:
            self.use_woe = False

        return self

    def transform(self, X):
        if not getattr(self, "use_woe", False):
            return X

        return self.woe.transform(X)


pipeline = Pipeline([
    (
        "aggregate-features",
        aggregateFeatures()
    ),
    (
        "date-features",
        DateFeatures()
    ),
    (
        "preprocessor",
        preprocessor
    )
    ,
    ("woe", WOETransformer())
])


