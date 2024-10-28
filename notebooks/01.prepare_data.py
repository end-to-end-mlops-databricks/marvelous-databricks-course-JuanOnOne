# COMMAND ----------
# !pip install ../dist/mlops_with_databricks-0.0.1-py3-none-any.whl -q

# Databricks notebook source
from mlops_with_databricks.preprocessor import DataProcessor
from mlops_with_databricks.config import ProjectConfig
import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# COMMAND ----------
filePath = "/Volumes/mlops_students/jarb8926/data/bank-additional-full.csv"
data_processor = DataProcessor(filePath, config, spark)

# Basic Data Prep
data_processor.prepare_data()

# COMMAND ----------
# Train/Test Split
train_data, test_data = data_processor.split_data()

# COMMAND ----------
# Save untransformed data splits
data_processor.save_to_catalog(train_data, test_data, spark)

