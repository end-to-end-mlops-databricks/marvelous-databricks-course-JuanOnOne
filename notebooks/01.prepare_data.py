# COMMAND ----------
# MAGIC %pip install ./mlops_with_databricks-0.0.1-py3-none-any.whl -q

# Databricks notebook source
from pyspark.sql import SparkSession

from mlops_with_databricks.config import ProjectConfig
from mlops_with_databricks.preprocessor import DataProcessor

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# COMMAND ----------
filePath = "/Volumes/mlops_students/jarb8926/data/bank-additional-full.csv"
data_processor = DataProcessor.from_file(filePath, config, spark)

# Basic Data Prep
data_processor.prepare_data()


# COMMAND ----------
# Save data to catalog
data_processor.save_raw_to_catalog()
data_processor.save_transformed_to_catalog()

# COMMAND ----------
