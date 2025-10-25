import pandas as pd
from dotenv import load_dotenv
import boto3
import os
import numpy as np
from difflib import get_close_matches
from io import StringIO
import requests
import pandas as pd
from datetime import datetime
import warnings
import json

warnings.simplefilter(action='ignore', category=Warning)

# Historic files https://drive.google.com/drive/folders/1UkCzBmz6YhaU8IKC0AuWHqGEUAMnVcBr
METRICS_FOLDER = "reporting/metrics"

now = datetime.now()
formatted_date = now.strftime("%m_%Y")

FINAL_METRIC_FOLDER = f"{METRICS_FOLDER}/{formatted_date}"

valid_neighborhoods = [
        'ALMAGRO', 'BALVANERA', 'BELGRANO', 'CABALLITO', 'COLEGIALES', 'DEVOTO',
        'FLORES', 'MONTSERRAT', 'NUNEZ', 'PALERMO', 'PARQUE PATRICIOS', 'PUERTO MADERO',
        'RECOLETA', 'RETIRO', 'SAN NICOLAS', 'SAN TELMO', 'VILLA CRESPO', 'VILLA DEL PARQUE', 'VILLA URQUIZA'
    ]

# Sección Utils
def clean_data(df):
    df_to_clean = df.copy(deep=True)

    try:
        res = requests.get("https://dolarapi.com/v1/dolares/blue")
        VALOR_DOLAR = res.json()["compra"]
    except Exception as e:
        return

    # En "location" tenemos la ubicación y el barrio. Separamos los valores en dos columnas
    df_to_clean[['neighborhood', 'location']] = df_to_clean['location'].str.split(",", expand=True)
    # Vamos a limpiar el valor de las expensas para que nos quede únicamente un número
    df_to_clean['expenses'] = df_to_clean['expenses'].str.extract(r'(\d{1,3}(?:\.\d{3})*)')
    # Vamos a normalizar el precio a una única moneda (Pesos)
    # Primero separaremos el valor en dos columnas, una para la moneda y otra para el valor
    df_to_clean[['price_currency', 'price']] = df_to_clean['price'].str.split(" ", expand=True)
    # Normalizamos a pesos
    df_to_clean = df_to_clean.loc[df_to_clean['price'].str.strip() != "precio"]
    # Oficial Venta
    df_to_clean.loc[df_to_clean['price_currency'] == 'USD', 'price'] = df_to_clean['price'].str.replace(".", "", regex=False).astype(float) * VALOR_DOLAR

    # Drop de price_currency
    df_to_clean.drop(columns=['price_currency'], inplace=True)

    # Extraemos las features y las separamos en distintas columnas
    total_area_pattern = r"(\d+)\s?m²"
    rooms_pattern = r"(\d+)\s?amb\.?"
    bedrooms_pattern = r"(\d+)\s?dorm\.?"
    bathrooms_pattern = r"(\d+)\s?bañ(?:os|o)"
    garages_pattern = r"(\d+)\s?coch\.?"

    # Extract values into new columns
    df_to_clean["total_area"] = df_to_clean["features"].str.extract(total_area_pattern, expand=True)
    df_to_clean["rooms"] = df_to_clean["features"].str.extract(rooms_pattern, expand=True)
    df_to_clean["bedrooms"] = df_to_clean["features"].str.extract(bedrooms_pattern, expand=True)
    df_to_clean["bathrooms"] = df_to_clean["features"].str.extract(bathrooms_pattern, expand=True)
    df_to_clean["garages"] = df_to_clean["features"].str.extract(garages_pattern, expand=True)


    # Convert extracted columns to numeric types
    df_to_clean[["total_area", "rooms", "bedrooms", "bathrooms", "garages"]] = (
        df_to_clean[["total_area", "rooms", "bedrooms", "bathrooms", "garages"]]
        .apply(pd.to_numeric)
    )

    df_to_clean.drop(columns=['features'], inplace=True)

    # Agregamos la columna antiguedad
    df_to_clean['antiquity'] = np.nan
    
    # Hay varias columnas que para la porción de ML no nos interesan. Las vamos a borrar
    try:
        df_to_clean.drop(columns=["location", "created_at", "updated_at", "property_url", "address"]
                , inplace=True)
    except KeyError:
        df_to_clean.drop(columns=["location", "property_url", "address"]
                , inplace=True)


    df_to_clean["price"] = df_to_clean["price"].str.replace(".000", "000").replace(".0", "")
    df_to_clean["price"] = df_to_clean["price"].str.replace(".0", "")

    # Convert extracted columns to numeric types
    df_to_clean[["total_area", "rooms", "bedrooms", "bathrooms", "garages", "price", "antiquity"]] = (
        df_to_clean[["total_area", "rooms", "bedrooms", "bathrooms", "garages", "price", "antiquity"]]
        .apply(pd.to_numeric, errors="coerce")
    )

    # Vamos a comenzar a llenar información faltante
    # Si la propiedad tiene NaN en garage, asumimos que no tiene
    df_to_clean.loc[df_to_clean["garages"].isna(), "garages"] = 0

    # Borramos las propiedades que no tienen precio
    df_to_clean = df_to_clean[~df_to_clean.price.isna()]

    # Borramos todas las propiedades que no tienen información de metros cuadrados
    df_to_clean = df_to_clean[~df_to_clean.total_area.isna()]

    # Vamos a trabajar ahora con numeros de cuartos, baños y ambientes
    # Pasamos las descripciones a lowercase para facilitar busquedas de strings
    df_to_clean["description"] = df_to_clean["description"].str.lower()

    # Si la propiedad menciona monoambiente en su descripción, asumimos que tiene 1 baño, 1 cuarto y 1 ambiente
    df_to_clean.loc[df_to_clean["description"].str.contains("monoambiente") & df_to_clean["bedrooms"].isna(), "bedrooms"] = 1
    df_to_clean.loc[df_to_clean["description"].str.contains("monoambiente") & df_to_clean["rooms"].isna(), "rooms"] = 1
    df_to_clean.loc[df_to_clean["description"].str.contains("monoambiente") & df_to_clean["bathrooms"].isna(), "bathrooms"] = 1

    # Vamos a seguir rellenando "rooms" en base a la descripción
    def rooms_filler(description):
        possible_rooms = [1,2,3,4,5,6,7,8,9,10]
        possible_descriptions = ["{} ambientes", "{} amb", "{} dormitorios", 
                                "{} dorm", "{} ambiente", "{}amb", "{} dor", "{}dorm", "{}  ambientes"]

        for i in possible_rooms:
            for j in possible_descriptions:
                if j.format(i) in description:
                    return i

    df_to_clean.loc[df_to_clean["rooms"].isna(), "rooms"] = df_to_clean.loc[df_to_clean["rooms"].isna(), "description"].apply(rooms_filler)

    # Eliminamos las propiedades que no tienen ambientes luego de este procesamiento
    df_to_clean = df_to_clean[~df_to_clean.rooms.isna()]

    # Realizamos conversores de baños y dormitorios

    # En resumen, si no tenemos cantidad de cuartos, asignamos la cantidad de cuartos menos 1
    # Sabemos que en este punto todos los registros tienen valor en rooms
    df_to_clean.loc[df_to_clean["bedrooms"].isna(), "bedrooms"] = df_to_clean.loc[df_to_clean["bedrooms"].isna()]["rooms"] - 1

    # En caso de que nos de 0, asumimos que es 1 ya que sería un monoambiente
    df_to_clean.loc[df_to_clean["bedrooms"] == 0, "bedrooms"] = 1

    def bathroom_converter(rooms):
        one_bathroom_values = [1,2,3]
        two_bathroom_values = [4,5,6]

        if rooms in one_bathroom_values:
            return 1
        elif rooms in two_bathroom_values:
            return 2
        else:
            return 3
        
    df_to_clean.loc[df_to_clean["bathrooms"].isna(), "bathrooms"] = df_to_clean.loc[df_to_clean["bathrooms"].isna()]["rooms"].apply(bathroom_converter)

    # Para antiguedad, si no tenemos valor, llenamos con el valor promedio del resto de las antiguedades
    # en el barrio
    df_to_clean['antiquity'] = df_to_clean['antiquity'].fillna(
        df_to_clean.groupby('neighborhood')['antiquity'].transform('mean')
    )

    # Round up to the nearest natural number
    df_to_clean['antiquity'] = np.ceil(df_to_clean['antiquity'])

    # Para muchos casos no tenemos antiguedad por promedio. Tomamos información de Internet y las rellenamos
    avg_antiquity = {
        "Recoleta": 50,  # Historical, many buildings from early 20th century
        "Núñez": 40,  # Mix of older houses and newer developments
        "Palermo Hollywood": 30,  # Many mid-century and newer constructions
        "Puerto Madero": 20,  # Mostly new developments since the 1990s
        "Centro / Microcentro": 70,  # Historic center with older buildings
        "Las Cañitas": 40,  # Trendy area with a mix of old and new
        "Palermo Soho": 40,  # Similar to Hollywood, slightly older buildings
        "Monte Castro": 50,  # Traditional residential area
        "Almagro Norte": 60,  # Older residential area
        "Tribunales": 80,  # Historic legal and business district
        "San Nicolás": 70,  # Similar to Microcentro
        "Monserrat": 80,  # One of the oldest neighborhoods
        "Belgrano R": 50,  # Residential, mix of old and newer homes
        "Palermo Nuevo": 30,  # Newer part of Palermo
        "Palermo Chico": 40,  # Upscale, many mid-century properties
        "Belgrano Chico": 40,  # Similar to Palermo Chico
        "Palermo Viejo": 50,  # Older buildings, many renovated
        "Retiro": 70,  # Historic with some modern developments
        "La Paternal": 50,  # Older residential neighborhood
        "Caballito Norte": 60,  # Older family homes
        "Belgrano C": 50,  # Similar to Belgrano R
        "Caballito Sur": 60,  # Same as Norte
        "Parque Rivadavia": 60,  # Older buildings near park
        "Villa Pueyrredón": 50,  # Traditional middle-class area
        "Floresta Sur": 60,  # Mix of old houses and mid-century buildings
        "Primera Junta": 60,  # Similar to Caballito
        "Cid Campeador": 60,  # Similar to surrounding areas
        "Constitución": 80,  # Old and densely built
        "Botánico": 40,  # Around the gardens, mix of styles
        "Lomas de Núñez": 30,  # Newer developments
        "Distrito Quartier": 20,  # Newly developed
        "Temperley": 70,  # Older suburb
        "Flores Sur": 60,  # Similar to Floresta
        "Almagro Sur": 60,  # Similar to Norte
        "Flores Norte": 60,  # Same as Sur
        "La Boca": 80,  # Historic with some modern projects
        "Parque Chas": 50,  # Traditional middle-class area
        "Floresta Norte": 60,  # Similar to Sur
        "Agronomía": 50,  # Near university, mix of styles
        "Otro": 50,  # Placeholder for undefined neighborhoods
        "Puerto Retiro": 70,  # Near historic Retiro
        "Barrio Parque": 40,  # Upscale, mid-century
        "Barrio Chino": 30,  # Newer commercial developments
        "Naón": 50,  # Mix of older homes
        "Parque Avellaneda": 60,  # Similar to Parque Chas
        "Catalinas": 20,  # Modern skyscrapers
        "Los Perales": 50,  # Traditional residential
        "Villa Riachuelo": 50,  # Outlying older area
        "Barrio Parque General Belgrano": 50  # Older homes, quieter
    }

    def antiquity_filler(neighborhood):
        return avg_antiquity.get(neighborhood, 50)

    df_to_clean.loc[df_to_clean["antiquity"].isna(), "antiquity"] = df_to_clean.loc[df_to_clean["antiquity"].isna()]["neighborhood"].apply(antiquity_filler)

    # Borramos la columna "expenses ya que no tenemos un buen uso por ahora"
    df_to_clean.drop(columns=["expenses"], inplace=True)

    # Borramos la descripción ya que no la vamos a usar en el modelo
    df_to_clean.drop(columns=["description"], inplace=True)

    # El precio tiene que ser mayor a 0 obligatoriamente
    df_to_clean = df_to_clean[df_to_clean["price"] > 1000]

    # El área total debe ser mayor a 0 obligatoriamente
    df_to_clean = df_to_clean[df_to_clean["total_area"] > 10]

    # Vemos que el precio tiene una distribución asimétrica a la derecha
    # Vamos a ahora quitar los outliers en relacion al precio

    # Calculamos los cuantiles
    Q1 = df_to_clean['price'].quantile(0.25)
    Q3 = df_to_clean['price'].quantile(0.75)
    IQR = Q3 - Q1

    # Calculamos los límites a partir de los cuantiles
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # identificamos los outliers y luego los borramos
    outliers = df_to_clean[(df_to_clean['price'] < lower_bound) | (df_to_clean['price'] > upper_bound)]

    # Remove outliers from the DataFrame
    df_to_clean = df_to_clean[(df_to_clean['price'] >= lower_bound) & (df_to_clean['price'] <= upper_bound)]

    # Tenemos un problema similar con la superficie total. Tomamos un approach similar

    # Vemos que el precio tiene una distribución asimétrica a la derecha
    # Vamos a ahora quitar los outliers en relacion al precio

    # Calculamos los cuantiles

    Q1 = df_to_clean['total_area'].quantile(0.25)
    Q3 = df_to_clean['total_area'].quantile(0.75)
    IQR = Q3 - Q1

    # Calculamos los límites a partir de los cuantiles
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # identificamos los outliers y luego los borramos
    outliers = df_to_clean[(df_to_clean['total_area'] < lower_bound) | (df_to_clean['total_area'] > upper_bound)]

    # Remove outliers from the DataFrame
    df_to_clean = df_to_clean[(df_to_clean['total_area'] >= lower_bound) & (df_to_clean['total_area'] <= upper_bound)]

    return df_to_clean

def extract_date(file):
    date_part = file.split('_')[-1].replace('.csv', '')  # Get "04022025"
    return pd.to_datetime(date_part, format="%d%m%Y")  # Convert to datetime

def map_neighborhood(neighborhood):

    # Lista de valores permitidos
    valid_neighborhoods = [
        'ALMAGRO', 'BALVANERA', 'BELGRANO', 'CABALLITO', 'COLEGIALES', 'DEVOTO',
        'FLORES', 'MONTSERRAT', 'NUNEZ', 'PALERMO', 'PARQUE PATRICIOS', 'PUERTO MADERO',
        'RECOLETA', 'RETIRO', 'SAN NICOLAS', 'SAN TELMO', 'VILLA CRESPO', 'VILLA DEL PARQUE', 'VILLA URQUIZA'
    ]

    neighborhood = neighborhood.upper()  # Convertimos a mayúsculas para estandarizar
    
    # Mapeos directos conocidos
    manual_mappings = {
        'MONSERRAT': 'MONTSERRAT',
        'NUÑEZ': 'NUNEZ',
        'CONGRESO': 'BALVANERA',
        'BARRIO NORTE': 'RECOLETA',
        'TRIBUNALES': 'SAN NICOLAS',
        'MICROCENTRO': 'SAN NICOLAS',
        'CENTRO / MICROCENTRO': 'SAN NICOLAS',
        'BARRACAS': 'PARQUE PATRICIOS',
        'CONSTITUCIÓN': 'SAN TELMO',
        'POMPEYA': 'PARQUE PATRICIOS',
        'MATADEROS': 'FLORES',
        'LINIERS': 'FLORES',
        'VERSALLES': 'VILLA URQUIZA',
        'VILLA SOLDATI': 'PARQUE PATRICIOS',
        'VILLA RIACHUELO': 'PARQUE PATRICIOS',
        'VILLA LUGANO': 'PARQUE PATRICIOS'
    }
    
    if neighborhood in manual_mappings:
        return manual_mappings[neighborhood]
    
    # Buscar coincidencias aproximadas
    match = get_close_matches(neighborhood, valid_neighborhoods, n=1, cutoff=0.6)
    return match[0] if match else 'OTRO'

load_dotenv()
session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY_ID'),
    region_name='us-east-2'
)
s3_client = session.client('s3')
BUCKET_NAME = os.getenv('BUCKET_NAME')

# Agarramos de zonaprop los dos archivos mas nuevos
files = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="data/raw/zonaprop")
sub_files = list()
for file in files["Contents"]:
    if ("data/raw/zonaprop" in file["Key"]) and (".csv" in file["Key"]):
        sub_files.append(file["Key"])

sorted_files = sorted(sub_files, key=extract_date, reverse=True)
newest_files = sorted_files[:2]

newest_file = newest_files[0]
previous_file = newest_files[1]

response = s3_client.get_object(Bucket=BUCKET_NAME, Key=newest_file)
csv_data = response['Body'].read().decode('utf-8')  # Convert bytes to string
df_new = pd.read_csv(StringIO(csv_data))

response = s3_client.get_object(Bucket=BUCKET_NAME, Key=previous_file)
csv_data = response['Body'].read().decode('utf-8')  # Convert bytes to string
df_old = pd.read_csv(StringIO(csv_data))

df_old = clean_data(df_old)
df_new = clean_data(df_new)

# Aplicar la función de mapeo
df_old['normalized_neighborhood'] = df_old['neighborhood'].apply(map_neighborhood)
df_new['normalized_neighborhood'] = df_new['neighborhood'].apply(map_neighborhood)

# Dropeamos los que tienen valor OTRO
df_old = df_old[df_old['normalized_neighborhood'] != 'OTRO']
df_new = df_new[df_new['normalized_neighborhood'] != 'OTRO']

# Borramos todos los contenidos dentro de la carpeta
print(f"Borrando contenido de la carpeta {FINAL_METRIC_FOLDER}")
objects_to_delete = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=FINAL_METRIC_FOLDER+"/")
if objects_to_delete["KeyCount"] > 0:
    s3_client.delete_objects(Bucket=BUCKET_NAME, Delete={'Objects': [{'Key': obj['Key']} for obj in objects_to_delete['Contents']]})
print(f"Contenido de la carpeta {FINAL_METRIC_FOLDER} borrado")

# Generamos los gráficos para cada barrio
for neighborhood in valid_neighborhoods:

    folder_name = f"{FINAL_METRIC_FOLDER}/{neighborhood}/"

    print(f"Generando metricas para el barrio {neighborhood}")

    import sys

    JSON_DATA = {}

    JSON_DATA["total_properties"] = len(df_new)

    if df_new[df_new["normalized_neighborhood"] == neighborhood].empty:
        JSON_DATA["total_properties_neighborhood"] = 0
        JSON_DATA["average_price_neighborhood"] = 0
        JSON_DATA["min_price_neighborhood"] = 0
        JSON_DATA["max_price_neighborhood"] = 0
    else:
        JSON_DATA["total_properties_neighborhood"] = len(df_new[df_new["normalized_neighborhood"] == neighborhood])
        JSON_DATA["average_price_neighborhood"] = str(df_new[df_new["normalized_neighborhood"] == neighborhood]["price"].mean().astype(int))
        JSON_DATA["min_price_neighborhood"] = str(df_new[df_new["normalized_neighborhood"] == neighborhood]["price"].min().astype(int))
        JSON_DATA["max_price_neighborhood"] = str(df_new[df_new["normalized_neighborhood"] == neighborhood]["price"].max().astype(int))

    JSON_STRING = json.dumps(JSON_DATA)
    s3_object_key = f"{FINAL_METRIC_FOLDER}/{neighborhood}/metrics.json"

    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_object_key,
        Body=JSON_STRING,
        ContentType='application/json'
    )
    print(f"JSON file uploaded successfully to s3://{BUCKET_NAME}/{s3_object_key}")