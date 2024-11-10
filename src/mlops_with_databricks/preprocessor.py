import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class DataProcessor:
    def __init__(self, filepath, config, spark: SparkSession):
        print(f"filepath: {filepath}")
        print(f"config: {config}")
        print(f"spark: {spark}")
        self.spark = spark
        self.df = self.load_data(filepath)
        self.config = config
        self.X = None
        self.y = None
        self.preprocessor = None
        self.target_encoder = None

    def load_data(self, filepath):
        df = self.spark.read.option("delimiter", ";").option("header", "true").csv(filepath).toPandas()
        return df

    def prepare_data(self):
        """Prepare the data for preprocessing"""

        # Handle numeric features
        num_features = self.config.num_features
        renamed_num_features = {col: col.replace(".", "_") for col in num_features if "." in col}
        if renamed_num_features:
            self.df.rename(columns=renamed_num_features, inplace=True)
            renamed_num_features = [v for _, v in renamed_num_features.items()]
        else:
            renamed_num_features = num_features
        for col in renamed_num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Convert categorical features from 'object' to 'category'
        cat_features = self.config.cat_features
        renamed_cat_features = {col: col.replace(".", "_") for col in cat_features if "." in col}
        if renamed_cat_features:
            self.df.rename(columns=renamed_cat_features, inplace=True)
            renamed_cat_features = [v for _, v in renamed_cat_features.items()]
        else:
            renamed_cat_features = cat_features
        for cat_col in renamed_cat_features:
            self.df[cat_col] = self.df[cat_col].astype("category")

        # Create primary key "Id", needed later for FeatureEngineering client.
        self.df["Id"] = pd.DataFrame(self.df.index + 1)

        # Extract target and relevant features
        target = self.config.target
        relevant_columns = renamed_cat_features + renamed_num_features + [target] + ["Id"]
        self.df = self.df[relevant_columns]

    def create_preprocessor(self):
        """Create the preprocessor without fitting"""
        num_features = self.config.num_features
        cat_features = self.config.cat_features

        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(max_categories=5, handle_unknown="ignore")),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, num_features), ("cat", categorical_transformer, cat_features)]
        )

        self.target_encoder = LabelEncoder()

    def split_data(self, test_size=0.2, random_state=42):
        """Split data for training."""
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state, stratify=self.y)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save splits to catalog."""

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
