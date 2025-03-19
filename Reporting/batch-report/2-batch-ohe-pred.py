import pandas as pd
import numpy as np
import joblib

# Cargar el modelo
MODEL_PATH = "models/model_ohe_13022025.joblib" #with one hot encoding
MODEL = joblib.load(MODEL_PATH)

# Cargar los datos
# df = pd.read_csv("ArgenpropData.csv")
df = pd.read_csv("output-simple-pred.csv") # pre-procesada por paso 1

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
    dt_ohe_data["total_area"] = float(row["total_area"])
    dt_ohe_data["rooms"] = float(row["rooms"])
    dt_ohe_data["bedrooms"] = float(row["bedrooms"])
    # dt_ohe_data["bathrooms"] = input_data.get("bathrooms", 0)  # Default to 0 if missing
    # dt_ohe_data["garages"] = input_data.get("garages", 0)  # Default to 0 if missing
    dt_ohe_data["antiquity"] = float(row["antiquity"])

    # Set one-hot encoding for the correct neighborhood
    neighborhood_col = "neighborhood_" + row["neighborhood"].upper()
    if neighborhood_col in dt_ohe_data:
        dt_ohe_data[neighborhood_col] = 1

    # Create the new DataFrame
    dt_ohe = pd.DataFrame([dt_ohe_data])

    input_features = np.array([
        dt_ohe
    ]).reshape(1, -1)
    return MODEL.predict(input_features)[0]


df["ohe_prediction"] = df.apply(predict, axis=1)

# Guardar el resultado
OUTPUT_PATH = "output-ohe-pred.csv"
df.to_csv(OUTPUT_PATH, index=False)

print(f"Procesamiento finalizado. Archivo guardado en {OUTPUT_PATH}")
