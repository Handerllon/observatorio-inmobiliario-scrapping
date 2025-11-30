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

#WARNING: por ahora esta construyendo solo el modelo con variable categórica (ohe)

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

#TODO: evaluar  response = s3_client.get_object(Bucket=BUCKET_NAME, Key=)
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
    """
        Modelo único para todo el dataset. Ej: CABA
    """
    df_ohe = pd.read_csv(LOCAL_DATAFILE)
    df_ohe.drop(columns=["bathrooms", "garages"], inplace=True)
    
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

# Columnas de barrios
NEIGHBORHOODS = [
    "neighborhood_ALMAGRO", "neighborhood_BALVANERA", "neighborhood_BELGRANO",
    "neighborhood_CABALLITO", "neighborhood_COLEGIALES", "neighborhood_DEVOTO",
    "neighborhood_FLORES", "neighborhood_MONTSERRAT", "neighborhood_NUNEZ",
    "neighborhood_PALERMO", "neighborhood_PARQUE PATRICIOS", "neighborhood_PUERTO MADERO",
    "neighborhood_RECOLETA", "neighborhood_RETIRO", "neighborhood_SAN NICOLAS",
    "neighborhood_SAN TELMO", "neighborhood_VILLA CRESPO", "neighborhood_VILLA DEL PARQUE",
    "neighborhood_VILLA URQUIZA"
]

def train_models_per_neighborhood():
    """
        Modelo por separado para cada barrio del dataset
        Se suben al bucket archivos diferentes para c/uno
    """
    df = pd.read_csv(LOCAL_DATAFILE)
    df.drop(columns=["bathrooms", "garages"], inplace=True)

    results = {}
    for neighborhood in NEIGHBORHOODS:
        #df_subset = df[df[neighborhood] == 1]
        df_subset = df[df[neighborhood] == 1].drop(columns=NEIGHBORHOODS)
        
        if df_subset.empty:
            print(f"No hay datos para {neighborhood}, omitiendo entrenamiento.")
            continue

        y = df_subset.price
        X = df_subset.drop(["price"], axis=1)
        # print(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        model = GradientBoostingRegressor(
            n_estimators=1000, learning_rate=0.01,
            max_depth=9, max_features='log2',
            min_samples_leaf=2, min_samples_split=5,
            loss='squared_error', random_state=5, subsample=0.5
        ).fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        rmse_score = mean_squared_error(y_test, y_pred, squared=False)
        model_score = model.score(X_test, y_test)
        
        model_filename = f"/tmp/model_ohe_{neighborhood}{LATEST}.joblib"
        joblib.dump(model, model_filename)
        print(f"Model saved locally: {model_filename}")
        
        s3_model_key = f"models/by-neighborhood/model_ohe_{neighborhood}{LATEST}.joblib"
        s3_client.upload_file(model_filename, BUCKET_NAME, s3_model_key)
        print(f"Model uploaded to s3://{BUCKET_NAME}/{s3_model_key}")
        
        results[neighborhood] = {
            "RMSE": rmse_score,
            "ModelScore": model_score,
            "S3ModelPath": f"s3://{BUCKET_NAME}/{s3_model_key}"
        }
    
    return results

def lambda_handler(event, context):
    result1 = train_ohe()
    result2 = train_models_per_neighborhood()

    # Combine both results into a single dictionary
    combined_result = {
        "OHEModel": result1,
        "NeighborhoodModels": result2
    }

    return {
        'statusCode': 200,
        'body': json.dumps(combined_result)
    }