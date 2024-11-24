# Databricks notebook source
# MAGIC %pip install ./mlops_with_databricks-0.0.1-py3-none-any.whl -q

# COMMAND ----------
# dbutils.library.restartPython()

# COMMAND ----------
import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession

from mlops_with_databricks.base_model import BaselineModel
from mlops_with_databricks.config import ProjectConfig
from mlops_with_databricks.preprocessor import DataProcessor
from mlops_with_databricks.utils import setup_logger

# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()


# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")
logger = setup_logger()

# Extract configuration details
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
processor = DataProcessor(config, spark)


# COMMAND ----------

############################
# Create Feature Table
############################

# Define feature table
logger.info("Create feature table.")
feature_table_name = f"{config.catalog_name}.{config.schema_name}.bank_marketing_features"

# Create or replace feature table
spark.sql(f"""
CREATE OR REPLACE TABLE {config.catalog_name}.{config.schema_name}.bank_marketing_features
(Id STRING NOT NULL,
 default_ STRING,
 housing STRING,
 loan STRING);
""")

# Add primary key to table
spark.sql(
    f"ALTER TABLE {config.catalog_name}.{config.schema_name}.bank_marketing_features "
    "ADD CONSTRAINT bank_marketing_pk PRIMARY KEY(Id);"
)

# enableChangeDataFeed to True
spark.sql(
    f"ALTER TABLE {config.catalog_name}.{config.schema_name}.bank_marketing_features "
    "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
)

# Insert data into the feature table from both train and test sets
spark.sql(
    f"INSERT INTO {config.catalog_name}.{config.schema_name}.bank_marketing_features "
    f"SELECT Id, housing, default, loan FROM {config.catalog_name}.{config.schema_name}.train_set"
)
spark.sql(
    f"INSERT INTO {config.catalog_name}.{config.schema_name}.bank_marketing_features "
    f"SELECT Id, housing, default, loan FROM {config.catalog_name}.{config.schema_name}.test_set"
)

# COMMAND ----------
############################
# Create FeatureFunction UDF
############################
function_name = f"{config.catalog_name}.{config.schema_name}.customer_risk"

# Define a function to calculate the customer risk of a possible default
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(default STRING, housing STRING, loan STRING)
RETURNS STRING
LANGUAGE PYTHON AS
$$
if default == 'unknown' or housing == 'unknown' or loan == 'unknown':
    return 'yes'
elif default == 'yes' or housing == 'yes' or loan == 'yes':
    return 'yes'
else:
    return 'no'
$$
""")

# COMMAND ----------
# Load training and test sets
train_set = DataProcessor.load_data_from_catalog("train_set", config, spark).df
train_set = train_set.drop("housing", "loan", "default")

# CAST primary key to "string"
train_set = train_set.withColumn("Id", train_set["Id"].cast("string"))

###########################
# Feature engineering setup
###########################
# Create training_set with feature_engineering client fe
logger.info("Create a training set...")
training_set = fe.create_training_set(
    df=train_set,
    label=config.target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["default", "housing", "loan"],
            lookup_key="Id",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="risk",
            input_bindings={"default": "default", "housing": "housing", "loan": "loan"},
        ),
    ],
    exclude_columns=["update_timestamp_utc"],
)

# Load feature-engineered DataFrame
training_df = training_set.load_df().toPandas()

# Calculate house_age for training and test set
test_set = DataProcessor.load_data_from_catalog("test_set", config, spark).df
test_set = test_set.selectExpr("*", f"{function_name}(default, housing, loan) as risk")
test_set = test_set.toPandas()

# COMMAND ----------
# Test Set
processor = DataProcessor.from_dataframe(test_set, config, spark)
X_test = processor.df
y_test = X_test[config.target]
X_test = X_test.drop(columns=[config.target, "update_timestamp_utc"])

# Train Set
processor = DataProcessor.from_dataframe(training_df, config, spark)
X_train = processor.df
y_train = X_train[config.target]
X_train = X_train.drop(columns=[config.target])

# Create Model
model = BaselineModel(config, spark)
processor.create_pipeline()
model.train(processor.pipeline.fit_transform(X_train), y_train)

# COMMAND ----------
##########
# MLFlow #
##########

# Set and start MLflow experiment
logger.info("Start experiment run...")
mlflow.set_experiment(experiment_name="/Shared/bank-marketing-fe")
git_sha = "d550d17a6c4cf5c5bd197d5daaa69fa3ad154d97"

with mlflow.start_run(tags={"branch": "week2", "git_sha": f"{git_sha}"}) as run:
    run_id = run.info.run_id

    # Pipeline fit/predict
    logger.info("## Instantiate, train and evaluate model...")
    model = BaselineModel(config, spark)
    processor.create_pipeline()
    model.train(processor.pipeline.fit_transform(X_train), y_train)
    metrics = model.evaluate(processor.pipeline.transform(X_test), y_test)
    y_pred = model.predict(processor.pipeline.transform(X_test))

    # Log model parameters, metrics, and model
    logger.info("## Log model parameters, metrics, and model...")
    mlflow.log_param("model_type", "RandomForestClassifier with preprocessing")
    mlflow.log_params(config.parameters)
    mlflow.log_metrics(metrics)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log model in experiments with feature engineering client fe
    logger.info("Log model...")
    fe.log_model(
        model=model.model,
        flavor=mlflow.sklearn,
        artifact_path="randomforest-pipeline-model-fe",
        training_set=training_set,
        signature=signature,
    )

# Register Model
logger.info("Register model...")
mlflow.register_model(
    model_uri=f"runs:/{run_id}/randomforest-pipeline-model-fe",
    name=f"{config.catalog_name}.{config.schema_name}.bank_marketing_model_fe",
)

# COMMAND ----------
