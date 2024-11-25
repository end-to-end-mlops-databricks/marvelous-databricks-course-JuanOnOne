from typing import Dict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from .config import ProjectConfig
from .preprocessor import DataProcessor


class BaselineModel:
    def __init__(self, config: ProjectConfig, processor: DataProcessor):
        self.config = config
        self.processor = processor
        self.model = RandomForestClassifier(**config.parameters)

    def __repr__(self):
        return f"BaselineModel(config='{self.config}', model={self.model})"

    def _load_data(self, dataset_name: str) -> tuple:
        processor = DataProcessor.load_data_from_catalog(dataset_name, self.config, self.processor)
        df = processor.df.toPandas()
        X = df.drop(columns=[self.config.target, "update_timestamp_utc"])
        y = df[self.config.target]
        return X, y

    def train(self, X_train: pd.DataFrame = None, y_train: pd.Series = None):
        if X_train is None or y_train is None:
            X_train, y_train = self._load_data("transformed_train_set")
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame = None) -> pd.Series:
        if X_test is None:
            X_test, _ = self._load_data("transformed_test_set")
        return self.model.predict(X_test)

    def evaluate(self, X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Dict[str, float]:
        if X_test is None or y_test is None:
            X_test, y_test = self._load_data("transformed_test_set")

        y_pred = self.predict(X_test)

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_pred),
        }

        for metric, value in metrics.items():
            print(f"{metric}: {value}")

        return metrics

    def get_feature_importance(self):
        feature_importance = self.model.named_steps["classifier"].feature_importances_
        feature_names = self.model.named_steps["preprocessor"].get_feature_names_out()
        return feature_importance, feature_names
