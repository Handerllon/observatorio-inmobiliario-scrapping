import json
import boto3
import joblib
import numpy as np
import os
import sklearn
import pandas as pd
print(sklearn.__version__)

"""
    Referencia: etl/utils.py

        input_data["total_area"]
        input_data["rooms"],
        input_data["bedrooms"],
        # input_data["bathrooms"],
        # input_data["garages"],
        input_data["antiquity"], 

IMPORTANTE:       
        Recordar configurar BUCKET_NAME como variable de entorno de la lambda
        y attachar la policy para que la lambda pueda acceder al bucket

        aws iam create-policy --policy-name LambdaS3ModelsAccessPolicy --policy-document file://s3-lambda-models-policy.json

        Get Your Lambda Role Name:
        aws lambda get-function --function-name [NOMBRELAMBDA] --query 'Configuration.Role' --output text

        Attach the New Policy to Your Lambda Role:
        aws iam attach-role-policy --role-name [LAMBDA-ROLE] --policy-arn [POLICY-ARN]

"""

#localhost
# load_dotenv()
# session = boto3.Session(
#     aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
#     aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY_ID')
# )

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
    if ("models" in file["Key"]) and (".joblib" in file["Key"]) and ("_wo_" in file["Key"]):
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
    required_fields = [
            "antiquity",
            "bedrooms",
            # "garages",
            # "bathrooms",
            "rooms",
            "total_area"]
    for field in required_fields:
        if field not in input_data:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(input_data[field], int):
            raise ValueError(f"Field '{field}' must be an integer.")

def predict(input_data):
    validate_input(input_data)
    input_features = np.array([
        input_data["total_area"],
        input_data["rooms"],
        input_data["bedrooms"],
        # input_data["bathrooms"],
        # input_data["garages"],
        input_data["antiquity"]
    ]).reshape(1, -1)
    prediction = MODEL.predict(input_features)
    print(f"Prediction: {prediction}")
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
