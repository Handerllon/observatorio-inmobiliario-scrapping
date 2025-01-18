import datetime
import pandas as pd
import requests
import numpy as np

def log(level, message):
    """
    Logs a message with a given severity level (INFO, WARNING, ERROR).
    
    Args:
        level (str): The severity level ('INFO', 'WARNING', or 'ERROR').
        message (str): The message to be logged.
    """
    levels = ["INFO", "WARNING", "ERROR"]
    if level not in levels:
        raise ValueError(f"Invalid log level: {level}. Use one of {levels}.")
    
    # Generate the timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format the log message
    log_message = f"[{timestamp}] [{level}] {message}"
    
    # Print the message to the console
    print(log_message)

### TODO: Una vez que tengamos el entorno de AWS corriendo constantemente podemos
### ya sea enviarlo a una tabla de DynamoDB o a un bucket de S3
def generate_raw_file(in_path, out_path):
    df = pd.read_csv(in_path)
    log("INFO", f"Generating RAW file from {in_path}")
    log("INFO", f"Original data: {df.shape}")
    df.drop_duplicates(subset=['zonaprop_code'], keep='first', inplace=True)
    log("INFO", f"Unique data: {df.shape}")

    log("INFO", "STOCK - RAW file processing completed. Generating RAW file...")
    df.to_csv(out_path, index=False)

### TODO: Idem lo de arriba. Por ahora generamos archivos locales
def generate_stg_file(in_path, out_path):
    df = pd.read_csv(in_path)
    log("INFO", f"Generating STG file from {in_path}")

    try:
        res = requests.get("https://dolarapi.com/v1/dolares/blue")
        VALOR_DOLAR = res.json()["compra"]
    except Exception as e:
        log("ERROR", f"Error in fetching the dollar value: {e}")
        return

    # En "location" tenemos la ubicación y el barrio. Separamos los valores en dos columnas
    df[['neighborhood', 'location']] = df['location'].str.split(",", expand=True)
    # Vamos a limpiar el valor de las expensas para que nos quede únicamente un número
    df['expenses'] = df['expenses'].str.extract(r'(\d{1,3}(?:\.\d{3})*)')
    # Vamos a normalizar el precio a una única moneda (Pesos)
    # Primero separaremos el valor en dos columnas, una para la moneda y otra para el valor
    df[['price_currency', 'price']] = df['price'].str.split(" ", expand=True)
    # Normalizamos a pesos
    df = df.loc[df['price'].str.strip() != "precio"]
    # Oficial Venta
    df.loc[df['price_currency'] == 'USD', 'price'] = df['price'].str.replace(".", "", regex=False).astype(float) * VALOR_DOLAR

    # Drop de price_currency
    df.drop(columns=['price_currency'], inplace=True)

    # Extraemos las features y las separamos en distintas columnas
    total_area_pattern = r"(\d+)\s?m²"
    rooms_pattern = r"(\d+)\s?amb\.?"
    bedrooms_pattern = r"(\d+)\s?dorm\.?"
    bathrooms_pattern = r"(\d+)\s?bañ(?:os|o)"
    garages_pattern = r"(\d+)\s?coch\.?"

    # Extract values into new columns
    df["total_area"] = df["features"].str.extract(total_area_pattern, expand=True)
    df["rooms"] = df["features"].str.extract(rooms_pattern, expand=True)
    df["bedrooms"] = df["features"].str.extract(bedrooms_pattern, expand=True)
    df["bathrooms"] = df["features"].str.extract(bathrooms_pattern, expand=True)
    df["garages"] = df["features"].str.extract(garages_pattern, expand=True)


    # Convert extracted columns to numeric types
    df[["total_area", "rooms", "bedrooms", "bathrooms", "garages"]] = (
        df[["total_area", "rooms", "bedrooms", "bathrooms", "garages"]]
        .apply(pd.to_numeric)
    )

    df.drop(columns=['features'], inplace=True)

    # Agregamos la columna antiguedad
    df['antiquity'] = np.nan

    log("INFO", "RAW - STG file processing completed. Generating STG file...")
    df.to_csv(out_path, index=False)


    

