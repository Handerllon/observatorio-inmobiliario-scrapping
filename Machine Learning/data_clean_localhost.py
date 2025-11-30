import pandas as pd
import numpy as np
from dotenv import load_dotenv
import boto3
import os
from io import StringIO
from datetime import datetime
from difflib import get_close_matches

date = datetime.now().strftime('%d%m%Y')
CLEANED_FILE = f"full_stg_extract_cleaned_{date}.csv"
OUTPUT_FILE = "machine_learning/data/full/" + CLEANED_FILE

# Obtenemos los últimos archivos de cada una de las fuentes
def extract_date(file):
    date_part = file.split('_')[-1].replace('.csv', '')  # Get "04022025"
    return pd.to_datetime(date_part, format="%d%m%Y")  # Convert to datetime


# Empezamos por ZonaProp
zonaprop_file = "STG_ZonaProp_12022025.csv"
print("Using zonaprop file {}".format(zonaprop_file))

#date = extract_date(zonaprop_file)

argenprop_file = "STG_ArgenProp_13022025.csv"
print("Using argenprop file {}".format(argenprop_file))

#response = s3_client.get_object(Bucket=BUCKET_NAME, Key=zonaprop_file)
#csv_data = response['Body'].read().decode('utf-8')  # Convert bytes to string
df_zonaprop = pd.read_csv(zonaprop_file)

#response = s3_client.get_object(Bucket=BUCKET_NAME, Key=argenprop_file)
#csv_data = response['Body'].read().decode('utf-8')  # Convert bytes to string
df_argenprop = pd.read_csv(argenprop_file)

df = pd.concat([df_zonaprop, df_argenprop], ignore_index=True)

# Hay varias columnas que para la porción de ML no nos interesan. Las vamos a borrar
try:
    df.drop(columns=["location", "created_at", "updated_at", "property_url", "address", "argenprop_code", "zonaprop_code"]
            , inplace=True)
except KeyError:
    df.drop(columns=["location", "property_url", "address", "argenprop_code", "zonaprop_code"]
            , inplace=True)


df["price"] = df["price"].str.replace(".000", "000").replace(".0", "")
df["price"] = df["price"].str.replace(".0", "")

# Convert extracted columns to numeric types
df[["total_area", "rooms", "bedrooms", "bathrooms", "garages", "price", "antiquity"]] = (
    df[["total_area", "rooms", "bedrooms", "bathrooms", "garages", "price", "antiquity"]]
    .apply(pd.to_numeric, errors="coerce")
)

# Vamos a comenzar a llenar información faltante
# Si la propiedad tiene NaN en garage, asumimos que no tiene
df.loc[df["garages"].isna(), "garages"] = 0

# Borramos las propiedades que no tienen precio
df = df[~df.price.isna()]

# Borramos todas las propiedades que no tienen información de metros cuadrados
df = df[~df.total_area.isna()]

# Vamos a trabajar ahora con numeros de cuartos, baños y ambientes
# Pasamos las descripciones a lowercase para facilitar busquedas de strings
df["description"] = df["description"].str.lower()

# Si la propiedad menciona monoambiente en su descripción, asumimos que tiene 1 baño, 1 cuarto y 1 ambiente
df.loc[df["description"].str.contains("monoambiente") & df["bedrooms"].isna(), "bedrooms"] = 1
df.loc[df["description"].str.contains("monoambiente") & df["rooms"].isna(), "rooms"] = 1
df.loc[df["description"].str.contains("monoambiente") & df["bathrooms"].isna(), "bathrooms"] = 1

# Vamos a seguir rellenando "rooms" en base a la descripción
def rooms_filler(description):
    possible_rooms = [1,2,3,4,5,6,7,8,9,10]
    possible_descriptions = ["{} ambientes", "{} amb", "{} dormitorios", 
                             "{} dorm", "{} ambiente", "{}amb", "{} dor", "{}dorm", "{}  ambientes"]

    for i in possible_rooms:
        for j in possible_descriptions:
            if j.format(i) in description:
                return i

df.loc[df["rooms"].isna(), "rooms"] = df.loc[df["rooms"].isna(), "description"].apply(rooms_filler)

# Eliminamos las propiedades que no tienen ambientes luego de este procesamiento
df = df[~df.rooms.isna()]

# Realizamos conversores de baños y dormitorios

# En resumen, si no tenemos cantidad de cuartos, asignamos la cantidad de cuartos menos 1
# Sabemos que en este punto todos los registros tienen valor en rooms
df.loc[df["bedrooms"].isna(), "bedrooms"] = df.loc[df["bedrooms"].isna()]["rooms"] - 1

# En caso de que nos de 0, asumimos que es 1 ya que sería un monoambiente
df.loc[df["bedrooms"] == 0, "bedrooms"] = 1

def bathroom_converter(rooms):
    one_bathroom_values = [1,2,3]
    two_bathroom_values = [4,5,6]

    if rooms in one_bathroom_values:
        return 1
    elif rooms in two_bathroom_values:
        return 2
    else:
        return 3
    
df.loc[df["bathrooms"].isna(), "bathrooms"] = df.loc[df["bathrooms"].isna()]["rooms"].apply(bathroom_converter)

# Para antiguedad, si no tenemos valor, llenamos con el valor promedio del resto de las antiguedades
# en el barrio
df['antiquity'] = df['antiquity'].fillna(
    df.groupby('neighborhood')['antiquity'].transform('mean')
)

# Round up to the nearest natural number
df['antiquity'] = np.ceil(df['antiquity'])

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

df.loc[df["antiquity"].isna(), "antiquity"] = df.loc[df["antiquity"].isna()]["neighborhood"].apply(antiquity_filler)

# Borramos la columna "expenses ya que no tenemos un buen uso por ahora"
df.drop(columns=["expenses"], inplace=True)

# Borramos la descripción ya que no la vamos a usar en el modelo
df.drop(columns=["description"], inplace=True)

# Normalizamos la información de barrios para que sea más fácil de usar en el modelo
# Función para mapear los barrios
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
        'VILLA LUGANO': 'PARQUE PATRICIOS',
        'LA PATERNAL': 'VILLA CRESPO',
        'COGHLAN': 'BELGRANO',
        'LAS CAÑITAS': 'BELGRANO',
        'VILLA PUEYRREDÓN': 'VILLA URQUIZA',
        'BOTÁNICO': 'PALERMO',
        'BOEDO': 'ALMAGRO',
        'CHACARITA': 'VILLA CRESPO',
        'PALERMO HOLLYWOOD': 'PALERMO',
        'SAN CRISTOBAL': 'BALVANERA',
        'SAAVEDRA': 'NUNEZ',
        'AGRONOMÍA': 'VILLA DEL PARQUE',
        'BARRIO PARQUE': 'PALERMO',
        'ONCE': 'BALVANERA',
        'BARRIO CHINO': 'BELGRANO',
        'LOMAS DE NÚÑEZ': 'NUNEZ',
        'LA BOCA': 'SAN TELMO',
        'MONTE CASTRO': 'DEVOTO',
        'CID CAMPEADOR': 'VILLA CRESPO',
        'BARRIO PARQUE GENERAL BELGRANO': 'BELGRANO',
        'PRIMERA JUNTA': 'CABALLITO',
        'VELEZ SARSFIELD': 'FLORES',
        'PARQUE AVELLANEDA': 'FLORES',
        'DISTRITO QUARTIER': 'PUERTO MADERO',
        'TEMPERLEY': 'FLORES',  # No está en CABA, asignado al más cercano
        'CATALINAS': 'RETIRO',
        'LOS PERALES': 'FLORES',
        'NAÓN': 'VILLA DEL PARQUE',
        'PATERNAL': 'VILLA CRESPO',
        'BOCA': 'SAN TELMO',
        'CONSTITUCION': 'SAN TELMO',
        'DEPARTAMENTO EN VENTA EN CABALLITO': 'CABALLITO',
        'DEPARTAMENTO EN VENTA EN PALERMO': 'PALERMO',
        'AGRONOMIA': 'VILLA DEL PARQUE'
    }
    
    if neighborhood in manual_mappings:
        return manual_mappings[neighborhood]
    
    # Buscar coincidencias aproximadas
    match = get_close_matches(neighborhood, valid_neighborhoods, n=1, cutoff=0.6)
    return match[0] if match else 'OTRO'

df['neighborhood'] = df['neighborhood'].apply(map_neighborhood)
df = df.loc[df["neighborhood"] != "OTRO"]

# Enviamos info a un archivo csv para trabajar en el siguiente paso
csv_buffer = "cleaned_zonaprop_argenprop_{}.csv".format(date)
df.to_csv(csv_buffer, index=False)
print("Cleaned data saved to {}".format(csv_buffer))
#print("Uploading data to S3")
#s3_client.put_object(Bucket=BUCKET_NAME, Key=OUTPUT_FILE, Body=csv_buffer.getvalue())
