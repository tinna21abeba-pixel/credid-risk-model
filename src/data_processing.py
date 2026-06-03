
"""Data processing and feature engineering utilities."""

from typing import Any
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Try to import xverse WOE transformer; provide a safe fallback if unavailable
try:
    from xverse.transformer import WOE as XVERSE_WOE  # type: ignore
    _HAS_XVERSE = True
except Exception:
    XVERSE_WOE = None
    _HAS_XVERSE = False



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
    """Apply Weight of Evidence transformation when labels available.

    If the external `xverse` package is not installed or fitting fails,
    this transformer is a no-op and leaves inputs unchanged.
    """

    def fit(self, X, y=None):
        self.use_woe = False
        if y is None or not _HAS_XVERSE:
            return self

        try:
            self.woe = XVERSE_WOE()
            self.woe.fit(X, y)
            self.use_woe = True
        except Exception:
            self.use_woe = False

        return self

    def transform(self, X):
        if not getattr(self, "use_woe", False):
            return X
        return self.woe.transform(X)


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    """Convert numpy output from ColumnTransformer back to a pandas DataFrame

    This transformer expects to be placed immediately after a fitted
    `ColumnTransformer` instance (passed via `preprocessor`) so it can
    obtain feature names via `get_feature_names_out`.
    """

    def __init__(self, preprocessor: ColumnTransformer):
        self.preprocessor = preprocessor

    def fit(self, X, y=None):
        try:
            # preprocessor should already be fitted by pipeline ordering
            self.columns_ = list(self.preprocessor.get_feature_names_out())
        except Exception:
            # Fallback generic names
            if hasattr(X, "shape"):
                self.columns_ = [f"f_{i}" for i in range(X.shape[1])]
            else:
                self.columns_ = []
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns_)


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
    ),
    (
        "to_dataframe",
        DataFrameTransformer(preprocessor)
    ),
    ("woe", WOETransformer())
])

def process_transactions(df):

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    snapshot_date = (
        df["TransactionStartTime"].max()
        + pd.Timedelta(days=1)
    )

    rfm = df.groupby("CustomerId").agg(
        Recency=(
            "TransactionStartTime",
            lambda x: (
                snapshot_date - x.max()
            ).days
        ),
        Frequency=(
            "TransactionId",
            "count"
        ),
        Monetary=(
            "Amount",
            "sum"
        )
    ).reset_index()

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[
            ["Recency",
             "Frequency",
             "Monetary"]
        ]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42
    )

    rfm["Cluster"] = kmeans.fit_predict(
        rfm_scaled
    )

    cluster_summary = rfm.groupby(
        "Cluster"
    )[[
        "Recency",
        "Frequency",
        "Monetary"
    ]].mean()

    high_risk_cluster = (
        cluster_summary["Frequency"]
        .idxmin()
    )

    rfm["is_high_risk"] = (
        rfm["Cluster"] == high_risk_cluster
    ).astype(int)

    df = df.merge(
        rfm[
            ["CustomerId",
             "is_high_risk"]
        ],
        on="CustomerId",
        how="left"
    )

    return df



