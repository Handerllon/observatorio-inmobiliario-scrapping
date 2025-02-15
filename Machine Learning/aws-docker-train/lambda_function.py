import json
import boto3
#import sklearn
#print(sklearn.__version__)
import numpy as np
import os

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle
import joblib
import os
import pandas as pd #fail

import re

"""
IMPORTANTE:       
        Recordar configurar BUCKET_NAME como variable de entorno de la lambda
        y attachar la policy para que la lambda pueda acceder al bucket

        aws iam create-policy --policy-name LambdaS3ModelsAccessPolicy --policy-document file://s3-lambda-models-policy.json

        Get Your Lambda Role Name:
        aws lambda get-function --function-name [NOMBRELAMBDA] --query 'Configuration.Role' --output text

        Attach the New Policy to Your Lambda Role:
        aws iam attach-role-policy --role-name [LAMBDA-ROLE] --policy-arn [POLICY-ARN]

"""


#Obtenemos los últimos archivos de cada una de las fuentes
def extract_date(file):
    date_part = file.split('_')[-1].replace('.csv', '')  # Get "04022025"
    return pd.to_datetime(date_part, format="%d%m%Y")  # Convert to datetime

s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv('BUCKET_NAME')

files = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
sub_files = list()
for file in files["Contents"]:
    if ("machine_learning/data/ml_ready_ohe" in file["Key"]) and (".csv" in file["Key"]):
        sub_files.append(file["Key"])

sorted_files = sorted(sub_files, key=extract_date, reverse=True)
DATAFILE_KEY = sorted_files[:1][0]
LOCAL_DATAFILE = "/tmp/data.csv"
# Download the model from S3
print(f"Downloading from s3://{BUCKET_NAME}/{DATAFILE_KEY}")
s3_client.download_file(BUCKET_NAME, DATAFILE_KEY, LOCAL_DATAFILE)

# Usamos una expresión regular para capturar la secuencia de 8 dígitos antes de ".csv"
match = re.search(r"_(\d{8})\.csv$", DATAFILE_KEY)
if match:
    LATEST = match.group(1)
else:
    print("No se encontró una fecha en el nombre del archivo.")

MODEL_FILE_OUT_OHE = "/tmp/model_ohe_"+LATEST+".joblib"
# Define model output filename
S3_MODEL_KEY = f"models/model_ohe_{LATEST}.joblib"

def train_ohe():
    df_ohe = pd.read_csv(LOCAL_DATAFILE)
    y = df_ohe.price
    X = df_ohe.drop(["price"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    # Con resultados del CVGridSearch grande
    GBoost2 = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01,
                                    max_depth=9, max_features='log2',
                                    min_samples_leaf=2, min_samples_split=5,
                                    loss='squared_error', random_state = 5, subsample=0.5)
    GBoost_model2 = GBoost2.fit(X_train, y_train)

    # Model evaluation
    GBoost_pred2 = GBoost_model2.predict(X_test)
    rmse_score = mean_squared_error(y_test, GBoost_pred2, squared=False)
    model_score = GBoost_model2.score(X_test, y_test)

    # print("RMSE score is: " + str(mean_squared_error(y_test, GBoost_pred2, squared=False))) # function'root_mean_squared_error
    # print("Model score is: " + str(GBoost_model2.score(X_test, y_test)))


    # Vamos a persistir el modelo que mejor nos dió
    #dump(GBoost_model2, MODEL_FILE_OUT_OHE)

    # Save model to local file
    joblib.dump(GBoost_model2, MODEL_FILE_OUT_OHE)
    print(f"Model saved locally: {MODEL_FILE_OUT_OHE}")

    # Upload model to S3
    s3_client.upload_file(MODEL_FILE_OUT_OHE, BUCKET_NAME, S3_MODEL_KEY)
    print(f"Model uploaded to s3://{BUCKET_NAME}/{S3_MODEL_KEY}")

    return {
        "RMSE": rmse_score,
        "ModelScore": model_score,
        "S3ModelPath": f"s3://{BUCKET_NAME}/{S3_MODEL_KEY}"
    }

def lambda_handler(event, context):
    result = train_ohe()
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }