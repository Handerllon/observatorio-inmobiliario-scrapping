import json
import boto3
import joblib
import numpy as np
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

def load_model_from_local(model_file):
    try:
        model_path = os.path.join(os.environ['LAMBDA_TASK_ROOT'], model_file)
        return joblib.load(model_path)
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

# Load the model on Lambda cold start (once per instance lifecycle)
BUCKET_NAME = "hordia-dev-tests"
MODEL_FILE = "model_wo_ohe_2024-12-04.joblib"
MODEL_PATH = download_model_from_s3(BUCKET_NAME, MODEL_FILE)
MODEL = joblib.load(MODEL_PATH)

def validate_input(input_data):
    required_fields = ["antiquity", "bedrooms", "garages", "bathrooms", "rooms", "total_area"]
    for field in required_fields:
        if field not in input_data:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(input_data[field], int):
            raise ValueError(f"Field '{field}' must be an integer.")

def predict(input_data):
    validate_input(input_data)
    input_features = np.array([
        input_data["antiquity"],
        input_data["bedrooms"],
        input_data["garages"],
        input_data["bathrooms"],
        input_data["rooms"],
        input_data["total_area"]
    ]).reshape(1, -1)
    prediction = MODEL.predict(input_features)
    return prediction.tolist()  # Convert to list for JSON serialization

def lambda_handler(event, context):
    try:
        if event.get("httpMethod") == "POST":
            body = json.loads(event.get("body", "{}"))
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

