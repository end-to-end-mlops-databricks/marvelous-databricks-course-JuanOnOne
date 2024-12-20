# COMMAND ----------
# MAGIC %pip install ./mlops_with_databricks-0.0.1-py3-none-any.whl -q

# Databricks notebook source
import json
import subprocess

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession

from mlops_with_databricks.base_model import BaselineModel
from mlops_with_databricks.config import ProjectConfig
from mlops_with_databricks.preprocessor import DataProcessor
from mlops_with_databricks.utils import setup_logger

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")
mlflow_client = MlflowClient()
logger = setup_logger()

# COMMAND ----------

# Extract configuration details
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
processor = DataProcessor(config, spark)

# COMMAND ----------
# Find latest run id
run_id = mlflow.search_runs(
    experiment_names=["/Shared/bank-marketing"],
    filter_string="tags.branch='week2'",
).run_id[0]

# Load previous model
logger.info("Load original model...")
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
            predictions = {
                "Labels": prediction_labels,
                "Probabilities": prediction_probabilities,
            }
            return predictions
        else:
            raise ValueError("Input must be a pandas DataFrame.")


# COMMAND ----------
# Train/Test Splits
logger.info("Instantiate, train and evaluate model.")
model = BaselineModel(config, processor)
model.train()
metrics = model.evaluate()

logger.info("Prepare inputs for experiment run.")
train_set_spark = DataProcessor.load_data_from_catalog("transformed_train_set", config, spark).df
X_train = train_set_spark.toPandas()

# COMMAND ----------
logger.info("Test the pyfunc model on a single record...")
test_set_spark = DataProcessor.load_data_from_catalog("transformed_train_set", config, spark).df
X_test = test_set_spark.toPandas()
X_test = X_test.drop(columns=[config.target, "update_timestamp_utc"])
wrapped_model = CustomModelWrapper(model.model)  # we pass the loaded model to the wrapper
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
logger.info("Create mlflow experiment...")
mlflow.set_experiment(experiment_name="/Shared/bank-marketing-pyfunc")
git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()

# Start experiment
logger.info("Run experiment...")
with mlflow.start_run(tags={"branch": "week2", "git_sha": f"{git_sha}"}) as run:
    logger.info("Get run_id from current run...")
    run_id = run.info.run_id

    # Pull the schema
    logger.info("Get schema...")
    signature = infer_signature(model_input=X_train, model_output={"Prediction": example_prediction})

    # Log Training dataset
    logger.info("Create and log dataset...")
    dataset = mlflow.data.from_spark(
        train_set_spark,
        table_name=f"{config.catalog_name}.{config.schema_name}.transformed_train_set",
        version="0",
    )
    mlflow.log_input(dataset, context="training")

    # Add your custom code, notice code/*.whl
    logger.info("Setup conda environment with custom code...")
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            "code/mlops_with_databricks-0.0.1-py3-none-any.whl",
        ],
        additional_conda_channels=None,
    )

    # Log the model
    logger.info(f"Log pyfunc model in experiment run_id: {run_id}")
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-bank-marketing-model",
        code_paths=["./mlops_with_databricks-0.0.1-py3-none-any.whl"],
        signature=signature,
    )

# COMMAND ----------
# Register the model
logger.info("Register the pyfunc model...")
model_name = f"{config.catalog_name}.{config.schema_name}.bank_marketing_model_pyfunc"
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/pyfunc-bank-marketing-model",
    name=model_name,
    tags={"git_sha": f"{git_sha}"},
)
# COMMAND ----------
# Dump model information to a json
with open("model_version.json", "w") as json_file:
    json.dump(model_version.__dict__, json_file, indent=4)

# COMMAND ----------
# Create a model alias
logger.info("Create model alias...")
model_version_alias = "the_best_model"
mlflow_client.set_registered_model_alias(model_name, model_version_alias, "6")

logger.info("Load pyfunc model using alias...")
model_uri = f"models:/{model_name}@{model_version_alias}"
model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------
model_version_output = dict(mlflow_client.get_model_version_by_alias(model_name, model_version_alias))
logger.info(f"{model_version_output}")


# COMMAND ----------
