from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from datetime import datetime

"""
     PyCaret model API

     Run: uvicorn inference_api:app --reload
     Swagger test/doc: http://127.0.0.1:8000/docs

     Example:
      curl -X 'POST' \
        'http://127.0.0.1:8000/predict' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
        "antiquity": 20,
        "bedrooms": 1,
        "rooms": 2,
        "total_area": 60
      }'
"""

# from pycaret.regression import load_model, predict_model
# import pandas as pd
from joblib import load

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Inicializar la aplicación FastAPI
app = FastAPI()

# without one-hot-encoding
# dump(GBoost_model2, 'models/model_ohe_2024-12-01.joblib')
model = load('../models/model_wo_ohe_2024-12-04.joblib')
if model:
    logger.info("Model loaded successfully.")

# Definir la estructura de datos de entrada
class ModelInput(BaseModel):
    antiquity: int
    bedrooms: int
    garages: int
    bathrooms: int 
    rooms: int
    total_area: int

# Endpoint para realizar predicciones
@app.post("/predict")
def predict(input_data: ModelInput):
    try:
        # Convertir la entrada a un DataFrame
        data_input = pd.DataFrame([[
            input_data.antiquity, 
            input_data.bedrooms, 
            input_data.rooms, 
            input_data.total_area
        ]], columns=["antiquity", "bedrooms", "rooms", "total_area"])
        
        #pycaret
        prediction = round( (predict_model(model, data=data_input))['prediction_label'][0] )
        
        # Devolver la predicción como JSON
        logger.info("Prediction made successfully.")
        logger.info(prediction)
        return {"prediction": str(prediction)}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

# Endpoint de prueba
@app.get("/")
def root():
    logger.info("Root endpoint accessed.")
    return {"message": "Inference API is running successfully."}
