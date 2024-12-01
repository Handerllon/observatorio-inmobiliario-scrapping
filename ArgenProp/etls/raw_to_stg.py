import boto3
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import numpy as np

VALOR_DOLAR = 1027.5
TARGET_TABLE_NAME = "STG_ArgenProp"

# Con este modo, el script toma información de la tabla DynamoDB
INTAKE_MODE = False
SOURCE_TABLE_NAME = "RAW_ArgenProp"

# Con este modo, el script toma información de un archivo
# Utilizado para testing para evitar consultar la tabla DynamoDB en cada corrida
FILE_MODE = True
FILE_PATH = "tmp/argenprop_raw_extract_2024-11-26.csv"

def get_all_items_from_table(client, table_name):
    """
    Retrieves all items from the specified DynamoDB table.
    """
    items = []
    response = client.scan(TableName=table_name)

    # Collect the first batch of items
    items.extend(response['Items'])

    # Continue fetching items if there are more
    while 'LastEvaluatedKey' in response:
        response = client.scan(
            TableName=table_name,
            ExclusiveStartKey=response['LastEvaluatedKey']
        )
        items.extend(response['Items'])
    
    return items

def dynamodb_items_to_dataframe(items):
    """
    Converts a list of DynamoDB items into a Pandas DataFrame.
    """
    # Convert DynamoDB format to Python dictionaries
    data = [{k: list(v.values())[0] for k, v in item.items()} for item in items]
    
    # Create a Pandas DataFrame
    return pd.DataFrame(data)

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

def batch_write_items(client, table_name, items):
    request_items = {
        table_name: [
            {"PutRequest": {"Item": item}} for item in items
        ]
    }

    # Call batch_write_item
    response = client.batch_write_item(RequestItems=request_items)
    unprocessed_items = response.get('UnprocessedItems', {})
    
    if unprocessed_items:
        print("Some items were not processed. Retrying...")
        # Retry logic here if needed
        print(unprocessed_items)

    return response

# Preparamos la información para procesarla
if INTAKE_MODE:
    if not SOURCE_TABLE_NAME:
        raise ValueError("Please provide a TABLE_NAME if INTAKE_MODE is set to True.")
    print("Running in Intake Mode...")
    load_dotenv()
    session = boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY_ID'),
        aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
    )

    client = session.client('dynamodb')
    print("Fetching data from DynamoDB...")
    items = get_all_items_from_table(client, SOURCE_TABLE_NAME)
    print("Converting items to DataFrame...")
    df = dynamodb_items_to_dataframe(items)
    print("Data loaded successfully.")
    df.to_csv("tmp/argenprop_raw_extract_{}.csv".format(str(datetime.now().strftime("%Y-%m-%d"))), index=False)

elif FILE_MODE:
    if not FILE_PATH:
        raise ValueError("Please provide a FILE_PATH if FILE_MODE is set to True.")
    print("Running in File Mode...")
    print("Loading data from file...")
    df = pd.read_csv(FILE_PATH)
    print("Data loaded successfully.")

else:
    raise ValueError("Please set either INTAKE_MODE or FILE_MODE to True.")

### Procesamos la información ###
print("Processing data...")

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

print("Data processed successfully")
print("Starting upload to table {}...".format(TARGET_TABLE_NAME))

if not INTAKE_MODE:
    load_dotenv()
    session = boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY_ID'),
        aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
    )
    client = session.client('dynamodb')

df_data = df.to_dict(orient='records')
processed_df_data = list()
for data in df_data:
    processed_df_data.append({
        'property_url': {'S': str(data['property_url'])},
        'argenprop_code': {'S': str(data['argenprop_code'])},
        'price': {'S': str(data['price'])},
        'expenses': {'S': str(data['expenses'])},
        'address': {'S': str(data['address'])},
        'location': {'S': str(data['location'])},
        'neighborhood': {'S': str(data['neighborhood'])},
        'description': {'S': str(data['description'])},
        'total_area': {'S': str(data['total_area'])},
        'rooms': {'S': str(data['rooms'])},
        'bedrooms': {'S': str(data['bedrooms'])},
        'bathrooms': {'S': str(data['bathrooms'])},
        'garages': {'S': str(data['garages'])},
        'antiquity': {'S': str(data['antiquity'])},
        'created_at': {'S': datetime.now().strftime('%Y-%m-%d')},
        'updated_at': {'S': datetime.now().strftime('%Y-%m-%d')},
    })

# Dividimos en chunks de 25 para no superar el límite de DynamoDB
chunks = [processed_df_data[i:i + 25] for i in range(0, len(processed_df_data), 25)]

n=1
for chunk in chunks:
    print(f"Uploading chunk {n} of {len(chunks)}")
    n+=1
    response = batch_write_items(client, TARGET_TABLE_NAME, chunk)