# COMMAND ----------
# MAGIC %pip install ./mlops_with_databricks-0.0.1-py3-none-any.whl -q

# COMMAND ----------
import subprocess

import mlflow
from mlflow.models import infer_signature
from pyspark.sql import SparkSession

from mlops_with_databricks.base_model import BaselineModel
from mlops_with_databricks.config import ProjectConfig
from mlops_with_databricks.preprocessor import DataProcessor
from mlops_with_databricks.utils import setup_logger

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")  # It must be -uc for registering models to Unity Catalog
logger = setup_logger()

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
config = ProjectConfig.from_yaml("../project_config.yml")
processor = DataProcessor(config, spark)

# Instantiate, train and evaluate model
logger.info("Instantiate, train and evaluate model.")
model = BaselineModel(config, processor)
model.train()
metrics = model.evaluate()

# COMMAND ----------
# Create a new experiment
logger.info("Create experiment.")
mlflow.set_experiment(experiment_name="/Shared/bank-marketing")
git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()

logger.info("Prepare inputs for experiment run.")
train_set_spark = DataProcessor.load_data_from_catalog("transformed_train_set", config, spark).df
X_train_transformed = train_set_spark.toPandas()
y_pred = model.predict()

# Start an mlflow run to track the training process
logger.info("Start Experiment run.")
with mlflow.start_run(
    tags={"git_sha": f"{git_sha}", "branch": "week2"},
) as run:
    logger.info("Get run_id from current run...")
    run_id = run.info.run_id

    # Log parameters, metrics, and the model to mlflow registry
    logger.info("Log parameters and metrics.")
    mlflow.log_param("model_type", "RandomForestClassifier with preprocessing")
    mlflow.log_params(config.parameters)
    mlflow.log_metrics(metrics)  # metrics is the BaseModel metrics.

    logger.info("Log dataset.")
    dataset = mlflow.data.from_spark(
        train_set_spark, table_name=f"{config.catalog_name}.{config.schema_name}.transformed_train_set", version="0"
    )
    mlflow.log_input(dataset, context="training")

    # Model logging
    logger.info("Log Model")
    signature = infer_signature(model_input=X_train_transformed, model_output=y_pred)
    mlflow.sklearn.log_model(
        sk_model=model.model, artifact_path="randomclassifier-pipeline-model", signature=signature
    )


# COMMAND ----------
# Model registry
logger.info("Register model artifacts.")
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/randomclassifier-pipeline-model",
    name=f"{config.catalog_name}.{config.schema_name}.bank_marketing_model_basic",
    tags={"git_sha": f"{git_sha}"},
)

# COMMAND ----------
logger.info("Get model information extracting the run_id")
run = mlflow.get_run(run_id)
dataset_info = run.inputs.dataset_inputs[0].dataset
dataset_source = mlflow.data.get_source(dataset_info)
dataset_source.load()

# COMMAND ----------
