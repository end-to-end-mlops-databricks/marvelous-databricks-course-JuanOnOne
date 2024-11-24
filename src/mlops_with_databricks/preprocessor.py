from typing import Tuple

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from .config import ProjectConfig
from .utils import setup_logger

logger = setup_logger()


class DataProcessor:
    def __init__(self, config: ProjectConfig, spark: SparkSession):
        self.config = config
        self.spark = spark
        self._df = None
        self.pipeline = None
        self.target_encoder = LabelEncoder()

    @classmethod
    def from_file(cls, filepath: str, config: ProjectConfig, spark: SparkSession):
        processor = cls(config, spark)
        processor._df = spark.read.option("delimiter", ";").option("header", "true").csv(filepath).toPandas()
        return processor

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, config: ProjectConfig, spark: SparkSession):
        processor = cls(config, spark)
        processor._df = df
        return processor

    @classmethod
    def load_data_from_catalog(cls, table_name: str, config: ProjectConfig, spark: SparkSession):
        """Loads data from catalog and returns spark.DataFrame"""
        processor = cls(config, spark)
        processor._df = spark.table(f"{config.catalog_name}.{config.schema_name}.{table_name}")
        return processor

    @property
    def df(self):
        if self._df is None:
            raise ValueError("DataFrame has not been initialized. Use one of the class methods to load data.")
        return self._df

    def _rename_columns(self, features):
        renamed = {col: col.replace(".", "_") if "." in col else col for col in features}
        self._df = self._df.rename(columns=renamed)
        return list(renamed.values()) if renamed else features

    def prepare_data(self):
        self.config.num_features = self._rename_columns(self.config.num_features)
        self.config.cat_features = self._rename_columns(self.config.cat_features)

        for col in self.config.num_features:
            self._df[col] = pd.to_numeric(self._df[col], errors="coerce")

        for col in self.config.cat_features:
            self._df[col] = self._df[col].astype("category")

        # Create index to be able to use as a Feature Table later.
        self._df["Id"] = self._df.index + 1
        relevant_columns = self.config.num_features + self.config.cat_features + [self.config.target, "Id"]
        self._df = self._df[relevant_columns]

        # Encode target variable
        self._df[self.config.target] = self.target_encoder.fit_transform(self._df[self.config.target])

    def create_pipeline(self):
        numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(max_categories=5, handle_unknown="ignore")),
            ]
        )

        self.pipeline = ColumnTransformer(
            [
                ("num", numeric_transformer, self.config.num_features),
                ("cat", categorical_transformer, self.config.cat_features),
                ("pass", "passthrough", ["Id"]),
            ]
        )

    def transform_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Transform data using sklearn pipelines."""
        X_train, X_test, y_train, y_test = self.split_data()

        if not self.pipeline:
            self.create_pipeline()

        X_train_transformed = self.pipeline.fit_transform(X_train)
        X_test_transformed = self.pipeline.transform(X_test)

        feature_names = (
            self.pipeline.named_transformers_["num"].get_feature_names_out(self.config.num_features).tolist()
            + self.pipeline.named_transformers_["cat"].get_feature_names_out(self.config.cat_features).tolist()
            + self.pipeline.named_transformers_["pass"].get_feature_names_out().tolist()
        )

        logger.info(f"Features: {feature_names}")

        X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)
        y_train_df = pd.DataFrame({self.config.target: y_train, "Id": X_train["Id"]}, index=y_train.index)
        y_test_df = pd.DataFrame({self.config.target: y_test, "Id": X_test["Id"]}, index=y_test.index)

        # Add target variable
        X_train_df = X_train_df.merge(y_train_df, on="Id", how="inner")
        X_test_df = X_test_df.merge(y_test_df, on="Id", how="inner")

        return X_train_df, X_test_df

    def split_data(self, test_size=0.2, random_state=42):
        """Split data."""
        X = self._df.drop(columns=[self.config.target])
        y = self._df[self.config.target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    def save_to_catalog(self, df: pd.DataFrame, table_name: str):
        spark_df = self.spark.createDataFrame(df).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        spark_df.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.{table_name}"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.{table_name} "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

    def save_raw_to_catalog(self):
        X_train, X_test, y_train, y_test = self.split_data()
        self.save_to_catalog(pd.concat([X_train, y_train], axis=1), "train_set")
        self.save_to_catalog(pd.concat([X_test, y_test], axis=1), "test_set")

    def save_transformed_to_catalog(self):
        train_df, test_df = self.transform_data()
        self.save_to_catalog(train_df, "transformed_train_set")
        self.save_to_catalog(test_df, "transformed_test_set")
