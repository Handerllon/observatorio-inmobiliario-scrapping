import json
import boto3
import joblib
import numpy as np
import os
import pandas as pd

import sklearn
print(sklearn.__version__)

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

# Obtenemos los últimos archivos de cada una de las fuentes
def extract_date(file):
    date_part = file.split('_')[-1].replace('.joblib', '')  # Get "04022025"
    return pd.to_datetime(date_part, format="%d%m%Y")  # Convert to datetime

s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv('BUCKET_NAME')
#BUCKET_NAME = "your-bucket-name"
# MODEL_KEY = "models/model_wo_ohe_13022025.joblib"  # Change as per your S3 structure
LOCAL_MODEL_PATH = "/tmp/last_model.joblib"  # Temporary local storage

files = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
sub_files = list()
for file in files["Contents"]:
    if ("models" in file["Key"]) and (".joblib" in file["Key"]) and ("_wo_" not in file["Key"]):
        sub_files.append(file["Key"])

sorted_files = sorted(sub_files, key=extract_date, reverse=True)
MODEL_KEY = sorted_files[:1][0]

# Download the model from S3
print(f"Downloading from s3://{BUCKET_NAME}/{MODEL_KEY}")
s3_client.download_file(BUCKET_NAME, MODEL_KEY, LOCAL_MODEL_PATH)

def load_model_from_local(model_file):
    try:
        print(f"Loading {LOCAL_MODEL_PATH} model")
        return joblib.load(LOCAL_MODEL_PATH)
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
    
MODEL = load_model_from_local(LOCAL_MODEL_PATH)


def validate_input(input_data):
    # Validate neighborhood as string
    if not isinstance(input_data["neighborhood"], str):
        raise ValueError("Field 'neighborhood' must be a string.")

    int_required_fields = [
            "antiquity",
            "bedrooms",
            # "garages",
            # "bathrooms",
            "rooms",
            "total_area"
    ]

    for field in int_required_fields:
        if field not in input_data:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(input_data[field], int):
            raise ValueError(f"Field '{field}' must be an integer.")

def predict(input_data):
    validate_input(input_data)
    
    ohe_columns = [
    #"price",
    "total_area",
    "rooms",
    "bedrooms",
    # "bathrooms",
    # "garages",
    "antiquity",
    "neighborhood_ALMAGRO", "neighborhood_BALVANERA", "neighborhood_BELGRANO",
    "neighborhood_CABALLITO", "neighborhood_COLEGIALES", "neighborhood_DEVOTO",
    "neighborhood_FLORES", "neighborhood_MONTSERRAT", "neighborhood_NUNEZ",
    "neighborhood_PALERMO", "neighborhood_PARQUE PATRICIOS", "neighborhood_PUERTO MADERO",
    "neighborhood_RECOLETA", "neighborhood_RETIRO", "neighborhood_SAN NICOLAS",
    "neighborhood_SAN TELMO", "neighborhood_VILLA CRESPO", "neighborhood_VILLA DEL PARQUE",
    "neighborhood_VILLA URQUIZA"
    ]

    # Initialize dictionary for DataFrame
    dt_ohe_data = {col: 0 for col in ohe_columns}

    # Assign values from input_data
    dt_ohe_data["total_area"] = float(input_data["total_area"])
    dt_ohe_data["rooms"] = float(input_data["rooms"])
    dt_ohe_data["bedrooms"] = float(input_data["bedrooms"])
    # dt_ohe_data["bathrooms"] = input_data.get("bathrooms", 0)  # Default to 0 if missing
    # dt_ohe_data["garages"] = input_data.get("garages", 0)  # Default to 0 if missing
    dt_ohe_data["antiquity"] = float(input_data["antiquity"])

    # Set one-hot encoding for the correct neighborhood
    neighborhood_col = "neighborhood_" + input_data["neighborhood"].upper()
    if neighborhood_col in dt_ohe_data:
        dt_ohe_data[neighborhood_col] = 1

    # Create the new DataFrame
    dt_ohe = pd.DataFrame([dt_ohe_data])

    print(f"Input array: {dt_ohe.iloc[0].to_dict()}")
    print(f"Using {MODEL_KEY} model")

    input_features = np.array([
        dt_ohe
    ]).reshape(1, -1)
    prediction = MODEL.predict(input_features)
    print(f"Prediction (ohe): {prediction}")
    return [str(prediction)] # Convert to list for JSON serialization

def lambda_handler(event, context):
    print(f"Received event: {json.dumps(event)}")  # Log the entire event

    # Assume POST if httpMethod is not present (for local testing)
    http_method = event.get("httpMethod", "POST")
    try:
        if http_method == "POST":
            # Handle body content
            if "body" in event and event["body"]:
                try:
                    body = json.loads(event["body"])  # Parse the body
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON format in body.")
            else:
                body = event  # Assume the event itself is the payload when body is absent
            print(f"Parsed body: {body}")  # Log the parsed body
            prediction = predict(body)
            return {
                "statusCode": 200,
                "body": json.dumps({"prediction": prediction}),
                "headers": {
                    "Content-Type": "application/json"
                }
            }
        else:
            return {
                "statusCode": 405,
                "body": json.dumps({"error": "Method not allowed. Use POST."}),
                "headers": {
                    "Content-Type": "application/json"
                }
            }
    except ValueError as ve:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(ve)}),
            "headers": {
                "Content-Type": "application/json"
            }
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
            "headers": {
                "Content-Type": "application/json"
            }
        }
