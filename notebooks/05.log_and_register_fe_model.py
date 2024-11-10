# Databricks notebook source
# MAGIC %pip install ./mlops_with_databricks-0.0.1-py3-none-any.whl

# COMMAND ----------
# dbutils.library.restartPython()

# COMMAND ----------
import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from mlops_with_databricks.config import ProjectConfig

# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.bank_marketing_features"

# COMMAND ----------
# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")

test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

# COMMAND ----------
############################
# Create Feature Table
############################

# Define feature table
feature_table_name = f"{catalog_name}.{schema_name}.bank_marketing_features"

# Create or replace feature table
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.bank_marketing_features
(Id STRING NOT NULL,
 default_ STRING,
 housing STRING,
 loan STRING);
""")

# Add primary key to table
spark.sql(
    f"ALTER TABLE {catalog_name}.{schema_name}.bank_marketing_features "
    "ADD CONSTRAINT bank_marketing_pk PRIMARY KEY(Id);"
)

# enableChangeDataFeed to True
spark.sql(
    f"ALTER TABLE {catalog_name}.{schema_name}.bank_marketing_features "
    "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
)

# Insert data into the feature table from both train and test sets
spark.sql(
    f"INSERT INTO {catalog_name}.{schema_name}.bank_marketing_features "
    f"SELECT Id, housing, default, loan FROM {catalog_name}.{schema_name}.train_set"
)
spark.sql(
    f"INSERT INTO {catalog_name}.{schema_name}.bank_marketing_features "
    f"SELECT Id, housing, default, loan FROM {catalog_name}.{schema_name}.test_set"
)

# COMMAND ----------

############################
# Create FeatureFunction UDF
############################
function_name = f"{catalog_name}.{schema_name}.customer_risk"

# Define a function to calculate the customer risk of a possible default
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(default_ STRING, housing STRING, loan STRING)
RETURNS STRING
LANGUAGE PYTHON AS
$$
if default_ == 'unknown' or housing == 'unknown' or loan == 'unknown':
    return 'yes'
elif default_ == 'yes' or housing == 'yes' or loan == 'yes':
    return 'yes'
else:
    return 'no'
$$
""")

# COMMAND ----------

# Load training and test sets
train_set = train_set.drop("housing", "loan", "default")

# CAST primary key to "string"
train_set = train_set.withColumn(
    "Id", train_set["Id"].cast("string")
)  # <- make sure "Id" it's string type!  # <- make sure "Id" it's string type!

###########################
# Feature engineering setup
###########################

# Create training_set with feature_engineering client fe
training_set = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["default_", "housing", "loan"],
            lookup_key="Id",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="risk",
            input_bindings={"default_": "default_", "housing": "housing", "loan": "loan"},
        ),
    ],
    exclude_columns=["update_timestamp_utc"],
)

# Load feature-engineered DataFrame
training_df = training_set.load_df().toPandas()
training_df.rename(columns={"default_": "default"}, inplace=True)

# Calculate house_age for training and test set
# test_set = spark.createDataFrame(test_set)
test_set = test_set.selectExpr("*", f"{function_name}(default, housing, loan) as risk")
test_set = test_set.toPandas()

# Split features and target
num_features = list(training_df.select_dtypes("number").columns)
cat_features = list(training_df.select_dtypes("category").columns) + ["risk"]
X_train = training_df[num_features + cat_features]
y_train = training_df[target]
X_test = test_set[num_features + cat_features]
y_test = test_set[target]

########################################
# Setup preprocessing and model pipeline
########################################
# Numeric Transformer
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

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
    transformers=[("num", numeric_transformer, num_features), ("cat", categorical_transformer, cat_features)],
    remainder="passthrough",
)

# Create the pipeline with preprocessing and the RandomForestClassifier
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", RandomForestClassifier(**parameters))])

##########
# MLFlow #
##########

# Set and start MLflow experiment
mlflow.set_experiment(experiment_name="/Shared/bank-marketing-fe")
git_sha = "ffa63b430205ff7"

with mlflow.start_run(tags={"branch": "week2", "git_sha": f"{git_sha}"}) as run:
    run_id = run.info.run_id

    # Pipeline fit/predict
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

    # Log model parameters, metrics, and model
    mlflow.log_param("model_type", "RandomForestClassifier with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc", auc)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log model in experiments with feature engineering client fe
    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path="randomforest-pipeline-model-fe",
        training_set=training_set,
        signature=signature,
    )

mlflow.register_model(
    model_uri=f"runs:/{run_id}/randomforest-pipeline-model-fe",
    name=f"{catalog_name}.{schema_name}.bank_marketing_model_fe",
)

# COMMAND ----------
