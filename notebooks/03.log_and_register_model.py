# COMMAND ----------
!pip install ./mlops_with_databricks-0.0.1-py3-none-any.whl

# Databricks notebook source

from pyspark.sql import SparkSession
from mlops_with_databricks.config import ProjectConfig
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
from mlflow.models import infer_signature

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri('databricks-uc') # It must be -uc for registering models to Unity Catalog

# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

# Load training and testing sets from Databricks tables
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

X_train = train_set[num_features + cat_features]
y_train = train_set[target]

X_test = test_set[num_features + cat_features]
y_test = test_set[target]

# COMMAND ----------

# Numeric Transformer
numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

# Categorical Transformer
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(max_categories=5, handle_unknown="ignore")),
    ]
)

# Label Encoder for target variable
le = LabelEncoder()

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, num_features), 
                  ('cat', categorical_transformer, cat_features)], 
    remainder='passthrough'
)

# Create the pipeline with preprocessing and the RandomForestClassifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestClassifier(**parameters))
])


# COMMAND ----------
mlflow.set_experiment(experiment_name='/Shared/bank-marketing')
git_sha = "ffa63b430205ff7"

# Start an MLflow run to track the training process
with mlflow.start_run(
    tags={"git_sha": f"{git_sha}",
          "branch": "week2"},
) as run:
    run_id = run.info.run_id

    pipeline.fit(X_train, le.fit_transform(y_train))
    y_test_encoded = le.transform(y_test)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model performance
    precision = precision_score(y_test_encoded, y_pred)
    recall = recall_score(y_test_encoded, y_pred)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    auc = roc_auc_score(y_test_encoded, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"AUC: {auc}")

    # Log parameters, metrics, and the model to MLFlow
    mlflow.log_param("model_type", "RandomForestClassifier with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc", auc)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    dataset = mlflow.data.from_spark(
    train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set",
    version="0")
    mlflow.log_input(dataset, context="training")
    
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="randomclassifier-pipeline-model",
        signature=signature
    )


# COMMAND ----------
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/randomclassifier-pipeline-model',
    name=f"{catalog_name}.{schema_name}.bank_marketing_model_basic",
    tags={"git_sha": f"{git_sha}"})

# COMMAND ----------
run = mlflow.get_run(run_id)
dataset_info = run.inputs.dataset_inputs[0].dataset
dataset_source = mlflow.data.get_source(dataset_info)
dataset_source.load()

# COMMAND ----------

