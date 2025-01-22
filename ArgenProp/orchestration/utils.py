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
    df.drop_duplicates(subset=['argenprop_code'], keep='first', inplace=True)
    log("INFO", f"Unique data: {df.shape}")

    log("INFO", "STOCK - RAW file processing completed. Generating RAW file...")
    df.to_csv(out_path, index=False)

def extract_total_area(values):
    tmp_values = eval(values)
    for value in tmp_values:
        if "m²" in value:
            return value.replace("m² cubie.", "").strip() 
    # Si no encontramos el valor. Devolvemos NaN
    return np.nan

def extract_rooms(values):
    tmp_values = eval(values)
    if "Monoam." in tmp_values:
        return str(1)
    for value in tmp_values:
        if "ambientes" in value:
            return value.replace("ambientes", "").strip()   
    # Si no encontramos el valor. Devolvemos NaN
    return np.nan

def extract_bedrooms(values):
    tmp_values = eval(values)
    for value in tmp_values:
        if "dorm" in value:
            return value.replace("dorm.", "").strip()   
    # Si no encontramos el valor. Devolvemos NaN
    return np.nan

def extract_bathrooms(values):
    tmp_values = eval(values)
    for value in tmp_values:
        if "baño" in value:
            return value.replace("baños", "").replace("baño", "").strip()   
    # Si no encontramos el valor. Devolvemos NaN
    return np.nan

def extract_antiquity(values):
    tmp_values = eval(values)
    if "A Estrenar" in tmp_values:
        return str(0)
    for value in tmp_values:
        if "años" in value:
            return value.replace("años", "").strip()   
    # Si no encontramos el valor. Devolvemos NaN
    return np.nan


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

    # En Location tenemos la ubicación y el barrio, vamos a separarlos
    df[['neighborhood', 'location']] = df['location'].str.split(",", expand=True)

    # Nos queda en la columna neighborhood el barrio con algo de texto extra. Removemos ese texto
    df['neighborhood'] = df['neighborhood'].str.replace("Departamento en Alquiler en", "").str.strip()
    df['location'] = df['location'].str.strip()

    # En ocasiones vemos que la columna location tiene en realidad el barrio. Vamos a corregir esto
    # para ser consistentes
    df.loc[df['location'] != "Capital Federal", 'neighborhood'] = df['location']

    # Luego de esto, reemplazamos los valores por capital federal. Tener en cuenta que el filtro ya viene
    # de la parte de scrapping
    df['location'] = "Capital Federal"

    # Vamos a limpiar el valor de las expensas para que nos quede un número
    df['expenses'] = df['expenses'].str.replace("$", "").str.replace(".", "").str.replace("expensas", "").str.strip()

    # Vamos a normalizar el precio a una única moneda (pesos argentinos)
    # Primero separaremos el valor en dos columnas, una para la moneda y otra para el valor
    df[['price_currency', 'price']] = df['price'].str.split(" ", expand=True)

    # Normalizamos a pesos
    # Tenemos algunas filas con valor None, las limpiamos
    df = df.loc[df['price'].notna()]
    df = df.loc[df['price'] != "None"]

    df.loc[df['price_currency'] == 'USD', 'price'] = df['price'].str.replace(".", "", regex=False).astype(float) * VALOR_DOLAR

    # Drop de price_currency
    df.drop(columns=['price_currency'], inplace=True)

    # Vamos a ahora extraer las features de la columna "features"

    df["total_area"] = df["features"].apply(extract_total_area)
    df["rooms"] = df["features"].apply(extract_rooms)
    df["bedrooms"] = df["features"].apply(extract_bedrooms)
    df["bathrooms"] = df["features"].apply(extract_bathrooms)
    df["antiquity"] = df["features"].apply(extract_antiquity)
    df["garages"] = np.nan

    df.drop(columns=['features'], inplace=True)
    
    log("INFO", "RAW - STG file processing completed. Generating STG file...")
    df.to_csv(out_path, index=False)


    

