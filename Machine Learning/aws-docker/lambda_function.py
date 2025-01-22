import json
import boto3
import joblib
import numpy as np
import os
import sklearn
print(sklearn.__version__)


def load_model_from_local(model_file):
    try:
        model_path = os.path.join(os.environ['LAMBDA_TASK_ROOT'], model_file)
        return joblib.load(model_path)
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
    
# Load the model on Lambda cold start (once per instance lifecycle)
#MODEL_FILE = "model_wo_ohe_2024-12-04.joblib" # 4 features
MODEL_FILE = "model_wo_ohe_2024-12-01.joblib"  # 6 features
MODEL = load_model_from_local(MODEL_FILE)

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
