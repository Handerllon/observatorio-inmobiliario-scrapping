import json
import boto3
import joblib
import os
import numpy as np
import pandas as pd


"""
    Referencia: etl/utils.py

        input_data["total_area"]
        input_data["rooms"],
        input_data["bedrooms"],
        # input_data["bathrooms"],
        # input_data["garages"],
        input_data["antiquity"], 
        input_data["neighborhood"] # String

IMPORTANTE:       
        Recordar configurar BUCKET_NAME como variable de entorno de la lambda
        y attachar la policy para que la lambda pueda acceder al bucket

        aws iam create-policy --policy-name LambdaS3ModelsAccessPolicy --policy-document file://s3-lambda-models-policy.json

        Get Your Lambda Role Name:
        aws lambda get-function --function-name [NOMBRELAMBDA] --query 'Configuration.Role' --output text

        Attach the New Policy to Your Lambda Role:
        aws iam attach-role-policy --role-name [LAMBDA-ROLE] --policy-arn [POLICY-ARN]

"""


# Configurar cliente de S3
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv('BUCKET_NAME')

# Obtener el modelo desde S3 según el barrio indicado en el input
def get_model_path(neighborhood):
    files = s3_client.list_objects_v2(Bucket=BUCKET_NAME).get("Contents", [])
    model_files = [file["Key"] for file in files if f"models/by-neighborhood/model_ohe_neighborhood_{neighborhood}" in file["Key"]]
    if not model_files:
        raise ValueError(f"No se encontró modelo para el barrio {neighborhood}. File: {model_files}")
    model_files.sort(reverse=True)
    return model_files[0]

# Descargar modelo desde S3
def load_model(neighborhood):
    model_key = get_model_path(neighborhood)
    local_model_path = f"/tmp/{os.path.basename(model_key)}"
    print(f"Downloading model from s3://{BUCKET_NAME}/{model_key}")
    s3_client.download_file(BUCKET_NAME, model_key, local_model_path)
    return joblib.load(local_model_path)

# Validar input
def validate_input(input_data):
    required_fields = ["total_area", "rooms", "bedrooms", "antiquity", "neighborhood"]
    for field in required_fields:
        if field not in input_data:
            raise ValueError(f"Missing required field: {field}")
        if field != "neighborhood" and not isinstance(input_data[field], int):
            raise ValueError(f"Field '{field}' must be an integer.")
    return input_data["neighborhood"].upper()

# Realizar predicción
def predict(input_data):
    neighborhood = validate_input(input_data)
    model = load_model(neighborhood)
    input_features = np.array([
        input_data["total_area"],
        input_data["rooms"],
        input_data["bedrooms"],
        input_data["antiquity"]
    ]).reshape(1, -1)
    prediction = model.predict(input_features)
    print(f"Prediction: {prediction}")
    return prediction.tolist()

# Handler de la Lambda
def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}")) if "body" in event else event
        prediction = predict(body)
        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": prediction}),
            "headers": {"Content-Type": "application/json"}
        }
    except ValueError as ve:
        return {"statusCode": 400, "body": json.dumps({"error": str(ve)}), "headers": {"Content-Type": "application/json"}}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)}), "headers": {"Content-Type": "application/json"}}
