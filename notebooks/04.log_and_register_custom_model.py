# COMMAND ----------
# MAGIC %pip install ./mlops_with_databricks-0.0.1-py3-none-any.whl

# Databricks notebook source
import json
import subprocess
import numpy as np
import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession

from mlops_with_databricks.config import ProjectConfig

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")
mlflow_client = MlflowClient()

# COMMAND ----------
config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target

parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
# Find latest run id
run_id = mlflow.search_runs(
    experiment_names=["/Shared/bank-marketing"],
    filter_string="tags.branch='week2'",
).run_id[0]

# Load previous model
model = mlflow.sklearn.load_model(f"runs:/{run_id}/randomclassifier-pipeline-model")


# COMMAND ----------
# Create a custom model that utilizies pyfunc.
# The predictions are a dictionary of labels and probabilities.
class CustomModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            prediction_labels = self.model.predict(model_input)[0]
            prediction_probabilities = np.max(self.model.predict_proba(model_input), axis=1)
            predictions = {"Labels": prediction_labels, "Probabilities": prediction_probabilities}
            return predictions
        else:
            raise ValueError("Input must be a pandas DataFrame.")


# COMMAND ----------
# Train/Test Splits
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

num_features = list(train_set.select_dtypes("number").columns)
cat_features = list(train_set.select_dtypes("category").columns)
X_train = train_set[num_features + cat_features]
y_train = train_set[[target]]

X_test = test_set[num_features + cat_features]
y_test = test_set[[target]]

# COMMAND ----------
wrapped_model = CustomModelWrapper(model)  # we pass the loaded model to the wrapper
example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
example_prediction = wrapped_model.predict(context=None, model_input=example_input)
print("Example Prediction:", example_prediction)

# COMMAND ----------
# this is a trick with custom packages
# https://docs.databricks.com/en/machine-learning/model-serving/private-libraries-model-serving.html
# but does not work with pyspark, so we have a better option :-)

##########
# MLFlow #
##########

# Create a new experiment
mlflow.set_experiment(experiment_name="/Shared/bank-marketing-pyfunc")
git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()

# Start experiment
with mlflow.start_run(tags={"branch": "week2", "git_sha": f"{git_sha}"}) as run:
    run_id = run.info.run_id
    # Pull the schema
    signature = infer_signature(model_input=X_train, model_output={"Prediction": example_prediction})
    # Log Training dataset
    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")
    # Add your code, notice code/*.whl
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            "code/mlops_with_databricks-0.0.1-py3-none-any.whl",
        ],
        additional_conda_channels=None,
    )

    # Log the model
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-bank-marketing-model",
        code_paths=["./mlops_with_databricks-0.0.1-py3-none-any.whl"],
        signature=signature,
    )

# COMMAND ----------
# Load Model and unwrap
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-bank-marketing-model")
loaded_model.unwrap_python_model()

# COMMAND ----------
# Register the model
model_name = f"{catalog_name}.{schema_name}.bank_marketing_model_pyfunc"
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/pyfunc-bank-marketing-model", name=model_name, tags={"git_sha": f"{git_sha}"}
)
# COMMAND ----------
# Dump model information to a json
with open("model_version.json", "w") as json_file:
    json.dump(model_version.__dict__, json_file, indent=4)

# COMMAND ----------
# Create a model alias
model_version_alias = "the_best_model"
mlflow_client.set_registered_model_alias(model_name, model_version_alias, "2")

model_uri = f"models:/{model_name}@{model_version_alias}"
model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------
model_version_output = dict(mlflow_client.get_model_version_by_alias(model_name, model_version_alias))
# COMMAND ----------
model

# COMMAND ----------
