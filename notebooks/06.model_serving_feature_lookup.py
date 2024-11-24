# Databricks notebook source
# MAGIC %pip install ./mlops_with_databricks-0.0.1-py3-none-any.whl

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Online Table for house features
# MAGIC We already created bank_marketing table as feature look up table.

# COMMAND ----------

import time

import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

from mlops_with_databricks.config import ProjectConfig
from mlops_with_databricks.utils import setup_logger

spark = SparkSession.builder.getOrCreate()
logger = setup_logger()

# Initialize Databricks clients
workspace = WorkspaceClient()
mlflow_client = MlflowClient()

# COMMAND ----------

# Load config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

######################
# Create online table
######################

# Create online table specification using OnlineTableSpec and OnlineTableSpecTriggeredSchedulingPolicy
logger.info("Creating OnlineTableSpec...")
spec = OnlineTableSpec(
    primary_key_columns=["Id"],
    source_table_full_name=f"{catalog_name}.{schema_name}.bank_marketing_features",
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=False,
)

# Create online table using spec
online_table_name = f"{catalog_name}.{schema_name}.bank_marketing_features_online"
try:
    online_table_pipeline = workspace.online_tables.create(name=online_table_name, spec=spec)
except Exception as e:
    if "already exists" in str(e):
        logger.info(f"Not creating online table {online_table_name} since it already exists.")
    else:
        raise e


# COMMAND ----------
######################
# Create endpoint    #
######################
logger.info("Creating model serving feature lookup endpoint...")

registered_model_name = f"{catalog_name}.{schema_name}.bank_marketing_model_fe"
versions = mlflow_client.search_model_versions(f"name='{registered_model_name}'")
registered_model_version = max(int(v.version) for v in versions)

endpoint_name = "bank-marketing-classifier"

try:
    status = workspace.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=registered_model_name,
                    entity_version=registered_model_version,
                    scale_to_zero_enabled=True,
                    workload_size="Small",
                )
            ]
        ),
    )
    logger.info(f"endpoint status:{status}")

except Exception as e:
    if "already exists" in str(e):
        logger.info(f"Not creating endpoint {endpoint_name} since it already exists.")
    else:
        raise e

# COMMAND ----------
# MAGIC %md
# MAGIC ### Call the endpoint

# COMMAND ----------
# Get token and host
logger.info("Get token and host to test endpoint...")

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()  # type: ignore # noqa: F821
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# Excluding "housing", "loan", "default" because they will be taken from feature look up
remove_columns = set(["housing", "loan", "default", "y"])
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
num_features = list(train_set.select_dtypes("number").columns)
cat_features = list(train_set.select_dtypes("object").columns)
required_columns = list(set(num_features + cat_features) - remove_columns)

# Construct dataframe_records to test endpoint
logger.info("Construct dataframe_records...")
sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------
train_set.dtypes

# COMMAND ----------
dataframe_records[0]

# COMMAND ----------
# POST request to endpoing using token and host.
logger.info("Testing endpoint...")
start_time = time.time()
model_serving_endpoint = f"https://{host}/serving-endpoints/bank-marketing-classifier/invocations"
response = requests.post(
    f"{model_serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records[0]},
)

end_time = time.time()
execution_time = end_time - start_time

logger.info(f"Response status: {response.status_code}")
logger.info(f"Response text:: {response.text}")
logger.info(f"Execution time: {execution_time} seconds")

# COMMAND ----------

bank_marketing_features = spark.table(f"{catalog_name}.{schema_name}.bank_marketing_features").toPandas()

# COMMAND ----------
bank_marketing_features.dtypes
