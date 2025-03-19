import pandas as pd
import numpy as np
import joblib

# Cargar el modelo
MODEL_PATH = "models/model_wo_ohe_13022025.joblib" #without one hot encoding
MODEL = joblib.load(MODEL_PATH)

# Cargar los datos
df = pd.read_csv("ArgenpropData.csv")

# Verificar que las columnas necesarias existen
required_columns = {"total_area", "rooms", "bedrooms", "antiquity"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Faltan columnas requeridas en el CSV: {required_columns - set(df.columns)}")

# Convertir las features a int
df["total_area"] = df["total_area"].astype(int)
df["rooms"] = df["rooms"].astype(int)
df["bedrooms"] = df["bedrooms"].astype(int)
df["antiquity"] = df["antiquity"].astype(int)

# Generar predicciones
def predict(row):
    # Revisar orden de las features con el modelo construido
    input_features = np.array([
        row["total_area"],
        row["rooms"],
        row["bedrooms"],
        row["antiquity"]
    ]).reshape(1, -1)
    return MODEL.predict(input_features)[0]

df["simple_prediction"] = df.apply(predict, axis=1)

# Guardar el resultado
OUTPUT_PATH = "output-simple-pred.csv"
df.to_csv(OUTPUT_PATH, index=False)

print(f"Procesamiento finalizado. Archivo guardado en {OUTPUT_PATH}")
