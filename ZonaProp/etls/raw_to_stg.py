import boto3
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import numpy as np

TARGET_TABLE_NAME = "STG_ZonaProp"
VALOR_DOLAR = 1027.5

# Con este modo, el script toma información de la tabla DynamoDB
INTAKE_MODE = False
SOURCE_TABLE_NAME = "RAW_ZonaProp"

# Con este modo, el script toma información de un archivo
# Utilizado para testing para evitar consultar la tabla DynamoDB en cada corrida
FILE_MODE = True
FILE_PATH = "tmp/zonaprop_raw_extract_2024-11-26.csv"

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
    df.to_csv("tmp/zonaprop_raw_extract_{}.csv".format(str(datetime.now().strftime("%Y-%m-%d"))), index=False)

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
        'zonaprop_code': {'S': str(data['zonaprop_code'])},
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