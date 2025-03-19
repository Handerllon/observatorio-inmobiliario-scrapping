import pandas as pd
import numpy as np
import joblib


# Cargar los datos
# df = pd.read_csv("ArgenpropData.csv")
df = pd.read_csv("output-ohe-pred.csv") #pre-procesada por paso 2

# Verificar que las columnas necesarias existen
required_columns = {"total_area", "rooms", "bedrooms", "antiquity","neighborhood"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Faltan columnas requeridas en el CSV: {required_columns - set(df.columns)}")

# Convertir las features a int
df["total_area"] = df["total_area"].astype(int)
df["rooms"] = df["rooms"].astype(int)
df["bedrooms"] = df["bedrooms"].astype(int)
df["antiquity"] = df["antiquity"].astype(int)

#str
df["neighborhood"] = df["neighborhood"].astype(str)

def predict(row):
    input_features = np.array([
        row["total_area"],
        row["rooms"],
        row["bedrooms"],
        row["antiquity"]
    ]).reshape(1, -1)

    # *********
    #WARNING!!!: deberia ser más eficiente ordenar por barrio antes de preprocesar para cargar cada modelo una sola vez. Con el argenprop code se podria recuperar el orden original o usar un index incremental!
    # o precargar todos los modelo e ir accediendolos según el caso (tipo dict)
    # *********

    # Cargar el modelo
    NEIGHBORHOOD = row["neighborhood"]
    MODELDATE = "15022025"
    MODEL_PATH = f"models/by-neighborhood/model_ohe_neighborhood_{NEIGHBORHOOD}{MODELDATE}.joblib" #with one hot encoding
    MODEL = joblib.load(MODEL_PATH)
    return MODEL.predict(input_features)[0]


df["neighborhood_prediction"] = df.apply(predict, axis=1)

# Guardar el resultado
OUTPUT_PATH = "output-final.csv"
df.to_csv(OUTPUT_PATH, index=False)

print(f"Procesamiento finalizado. Archivo guardado en {OUTPUT_PATH}")
