import pandas as pd
from dotenv import load_dotenv
import boto3
import os
from io import BytesIO, StringIO
from difflib import get_close_matches
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
from geopy.geocoders import Nominatim
import uuid
import warnings
warnings.simplefilter(action='ignore', category=Warning)

#Forces Matplotlib to store its cache in /tmp/, which is writable in AWS Lambda.
os.environ["MPLCONFIGDIR"] = "/tmp"

HISTORIC_FILE_PATH = "./alquilerescaba_202501.xlsx"

# Sección Utils
def clean_data(df):
    df_to_clean = df.copy(deep=True)
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

def extract_date(file):
    date_part = file.split('_')[-1].replace('.csv', '')  # Get "04022025"
    return pd.to_datetime(date_part, format="%d%m%Y")  # Convert to datetime

# Sección Upload
def upload_json_to_s3(bucket_name, key, data):
    s3_client.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=json.dumps(data),
        ContentType="application/json"
    )
    print(f"Uploaded JSON to s3://{bucket_name}/{key}")

def upload_image_to_s3(bucket_name, key, plt):
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close()
    img_buffer.seek(0)

    s3_client.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=img_buffer,
        ContentType="image/png"
    )
    return "https://{}.s3.us-east-2.amazonaws.com/{}".format(bucket_name, key)

# Sección Info estadística
def calculate_new_and_removed_properties_neighborhood(df_old, df_new, neighborhood=None):

    if not neighborhood:
        old_ids = df_old['zonaprop_code'].unique().tolist()
        new_ids = df_new['zonaprop_code'].unique().tolist()

        new_properties = len(df_new[~df_new['zonaprop_code'].isin(old_ids)])
        removed_properties = len(df_old[~df_old['zonaprop_code'].isin(new_ids)])

    else:
        old_ids = df_old[df_old['normalized_neighborhood'] == neighborhood]['zonaprop_code'].unique().tolist()
        new_ids = df_new[df_new['normalized_neighborhood'] == neighborhood]['zonaprop_code'].unique().tolist()

        new_properties = len(df_new[(df_new['normalized_neighborhood'] == neighborhood) & (~df_new['zonaprop_code'].isin(old_ids))])
        removed_properties = len(df_old[(df_old['normalized_neighborhood'] == neighborhood) & (~df_old['zonaprop_code'].isin(new_ids))])

    return new_properties, removed_properties

def generate_price_evolution_graph(bucket_name, folder_name, input_data):
    #Aca utilizamos la data de San Andres
    df = pd.read_excel(HISTORIC_FILE_PATH)
    # Quitamos del dataframe los locales y oficinas
    # Dataframe negar la condición
    df = df[~df['Inmueble'].isin(["Oficina", "Local"])]
    df.drop(columns=['Inmueble'], inplace=True)
    # Convertir la columna 'Mes' a formato datetime
    df['Mes'] = pd.to_datetime(df['Mes'], format='%Y-%m')
    df_grouped = df.groupby(['Mes', 'Barrio']).agg({'Mediana.por.m2.a.precios.corrientes': 'mean'}).reset_index()

    current_date = datetime.today()
    one_year_ago = current_date.replace(year=current_date.year - 1)
    formatted_date = one_year_ago.strftime('%Y-%m')

    df_grouped = df_grouped[df_grouped['Mes'] >= formatted_date]

    # Definir colores para cada zona
    colores = {
        input_data["neighborhood"]: "blue"
    }

    # Crear la figura y el gráfico
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    for zona, color in colores.items():
        subset = df_grouped[df_grouped['Barrio'] == zona]
        plt.plot(subset['Mes'], subset['Mediana.por.m2.a.precios.corrientes'], marker='o', label=zona, color=color)

    # Configurar etiquetas y título
    plt.xlabel("Mes")
    plt.ylabel("Precio por m2")
    plt.title("Evolución de Precios de Alquiler")
    plt.xticks(rotation=45)

    img_path = upload_image_to_s3(bucket_name, folder_name + "price_by_m2_evolution.png", plt)
    return img_path

def generate_bar_charts(df, bucket_name, folder_name):
    df_filtered = df

    # Agrupar las habitaciones en 1, 2, 3, y "4 o más"
    df_filtered['room_category'] = df_filtered['rooms'].apply(lambda x: x if x < 4 else '4 o más')
    df_filtered['price'] = pd.to_numeric(df_filtered['price'], errors='coerce')  # Convertir a número, manejando errores

    # Calcular el promedio de precio por cantidad de ambientes
    avg_price = df_filtered.groupby('room_category')['price'].mean()

    # Calcular el precio por metro cuadrado por cantidad de ambientes
    df_filtered['price_per_m2'] = df_filtered['price'] / df_filtered['total_area']
    avg_price_per_m2 = df_filtered.groupby('room_category')['price_per_m2'].mean()

    # Gráfico de barras - Promedio de precio por cantidad de ambientes
    plt.figure(figsize=(10, 5))
    ax = avg_price.plot(kind='bar', color='skyblue', edgecolor='black')
    # Agregar valores encima de cada barra
    for i, value in enumerate(avg_price):
        ax.text(i, value + (value * 0.02), f'{value:,.0f}', ha='center', fontsize=10, fontweight='bold')
    plt.title(f'Promedio de Precio por Cantidad de Ambientes para CABA')
    plt.xlabel('Cantidad de Ambientes')
    plt.ylabel('Precio Promedio ($)')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    img_path_1 = upload_image_to_s3(bucket_name, folder_name + "bar_price_by_amb.png", plt)


    # Gráfico de barras - Precio por metro cuadrado por cantidad de ambientes
    plt.figure(figsize=(10, 5))
    ax = avg_price_per_m2.plot(kind='bar', color='skyblue', edgecolor='black')
    # Agregar valores encima de cada barra
    for i, value in enumerate(avg_price_per_m2):
        ax.text(i, value + (value * 0.02), f'{value:,.0f}', ha='center', fontsize=10, fontweight='bold')
    plt.title('Precio por Metro Cuadrado por Cantidad de Ambientes para CABA')
    plt.xlabel('Cantidad de Ambientes')
    plt.ylabel('Precio por m² Promedio ($)')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    img_path_2 = upload_image_to_s3(bucket_name, folder_name + "bar_m2_price_by_amb.png", plt)
    return img_path_1, img_path_2

def generate_bar_charts_neighborhood(df, neighborhood, bucket_name, folder_name):
    df_filtered = df[df["normalized_neighborhood"] == neighborhood]

    # Agrupar las habitaciones en 1, 2, 3, y "4 o más"
    df_filtered['room_category'] = df_filtered['rooms'].apply(lambda x: x if x < 4 else '4 o más')
    df_filtered['price'] = pd.to_numeric(df_filtered['price'], errors='coerce')  # Convertir a número, manejando errores

    # Calcular el promedio de precio por cantidad de ambientes
    avg_price = df_filtered.groupby('room_category')['price'].mean()

    # Calcular el precio por metro cuadrado por cantidad de ambientes
    df_filtered['price_per_m2'] = df_filtered['price'] / df_filtered['total_area']
    avg_price_per_m2 = df_filtered.groupby('room_category')['price_per_m2'].mean()

    # Gráfico de barras - Promedio de precio por cantidad de ambientes
    plt.figure(figsize=(10, 5))
    ax = avg_price.plot(kind='bar', color='skyblue', edgecolor='black')
    # Agregar valores encima de cada barra
    for i, value in enumerate(avg_price):
        ax.text(i, value + (value * 0.02), f'{value:,.0f}', ha='center', fontsize=10, fontweight='bold')
    plt.title(f'Promedio de Precio por Cantidad de Ambientes para {neighborhood}')
    plt.xlabel('Cantidad de Ambientes')
    plt.ylabel('Precio Promedio ($)')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    img_path_1 = upload_image_to_s3(bucket_name, folder_name + "bar_price_by_amb_neighborhood.png", plt)

    # Gráfico de barras - Precio por metro cuadrado por cantidad de ambientes
    plt.figure(figsize=(10, 5))
    ax = avg_price_per_m2.plot(kind='bar', color='skyblue', edgecolor='black')
    # Agregar valores encima de cada barra
    for i, value in enumerate(avg_price_per_m2):
        ax.text(i, value + (value * 0.02), f'{value:,.0f}', ha='center', fontsize=10, fontweight='bold')
    plt.title('Precio por Metro Cuadrado por Cantidad de Ambientes para {}'.format(neighborhood))
    plt.xlabel('Cantidad de Ambientes')
    plt.ylabel('Precio por m² Promedio ($)')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    img_path_2 = upload_image_to_s3(bucket_name, folder_name + "bar_m2_price_by_amb_neighborhood.png", plt)
    return img_path_1, img_path_2

def generate_pie_charts(df, bucket_name, folder_name):
    df_filtered = df

    # Agrupar las habitaciones en 1, 2, 3 y "4 o más"
    df_filtered['room_category'] = df_filtered['rooms'].apply(lambda x: x if x < 4 else '4 o más')

    # Contar la cantidad de propiedades por categoría de ambientes
    room_distribution = df_filtered['room_category'].value_counts()

    # Función para mostrar tanto el porcentaje como la cantidad
    def autopct_format(pct, all_vals):
        absolute = int(round(pct/100.*sum(all_vals)))
        return f"{pct:.1f}%\n({absolute})"

    # Generar gráfico de torta (pie chart)
    plt.figure(figsize=(8, 8))
    plt.pie(
        room_distribution, 
        labels=room_distribution.index, 
        autopct=lambda pct: autopct_format(pct, room_distribution), 
        colors=['skyblue', 'lightcoral', 'gold', 'lightgreen'], 
        startangle=140
    )
    plt.title(f'Distribución de Propiedades por Cantidad de Ambientes CABA')
    img_path = upload_image_to_s3(bucket_name, folder_name + "pie_property_amb_distribution.png", plt)
    return img_path

def generate_pie_charts_neighborhood(df, neighborhood, bucket_name, folder_name):
    df_filtered = df[df["normalized_neighborhood"] == neighborhood]

    # Agrupar las habitaciones en 1, 2, 3 y "4 o más"
    df_filtered['room_category'] = df_filtered['rooms'].apply(lambda x: x if x < 4 else '4 o más')

    # Contar la cantidad de propiedades por categoría de ambientes
    room_distribution = df_filtered['room_category'].value_counts()

    # Función para mostrar tanto el porcentaje como la cantidad
    def autopct_format(pct, all_vals):
        absolute = int(round(pct/100.*sum(all_vals)))
        return f"{pct:.1f}%\n({absolute})"

    # Generar gráfico de torta (pie chart)
    plt.figure(figsize=(8, 8))
    plt.pie(
        room_distribution, 
        labels=room_distribution.index, 
        autopct=lambda pct: autopct_format(pct, room_distribution), 
        colors=['skyblue', 'lightcoral', 'gold', 'lightgreen'], 
        startangle=140
    )
    plt.title(f'Distribución de Propiedades por Cantidad de Ambientes {neighborhood}')
    img_path = upload_image_to_s3(bucket_name, folder_name + "pie_property_amb_distribution_neighborhood.png", plt)
    return img_path

# Sección lugares cercanos
def obtener_coordenadas(direccion):
    """ Convierte una dirección en coordenadas (latitud, longitud) usando Nominatim """
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.geocode(direccion)
    if location:
        return location.latitude, location.longitude
    return None

def consultar_overpass(lat, lon):
    """ Consulta la API de Overpass para obtener lugares cercanos """
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Radio de búsqueda (500 metros)
    radio = 500  

    # Hacemos una query buscando todos los elementos cercanos
    query = f"""
    [out:json];
    (
      node(around:{radio},{lat},{lon});
      way(around:{radio},{lat},{lon});
      relation(around:{radio},{lat},{lon});
    );
    out center;
    """
    
    response = requests.get(overpass_url, params={'data': query})
    if response.status_code == 200:
        return response.json()["elements"]
    return []

def obtener_lugares_cercanos(direccion):
    """ Obtiene los lugares cercanos de interés a una dirección dada """
    coordenadas = obtener_coordenadas(direccion)
    if not coordenadas:
        print("No se pudo obtener la ubicación.")
        return

    lat, lon = coordenadas

    resultados = {}
    datos = consultar_overpass(lat, lon)
    for lugar in datos:
        categoria = lugar.get("tags", {}).get("amenity", "Desconocido")
        nombre = lugar.get("tags", {}).get("name", "Sin nombre")
        lat = lugar.get("lat", 0)
        lon = lugar.get("lon", 0)
        if categoria not in resultados:
            resultados[categoria] = []
        resultados[categoria].append((nombre, lat, lon))

    return resultados

def procesar_resultados(lugares_cercanos):

    output_data = {
        "transporte": {"total": 0, "data": []},
        "sitios_interes": {"total": 0, "data": []},
        "edificios_administrativos": {"total": 0, "data": []},
        "instituciones_educativas": {"total": 0, "data": []},
        "centros_salud": {"total": 0, "data": []},
        "restaurantes": {"total": 0, "data": []}
    }

    for categoria, lugares in lugares_cercanos.items():
        if categoria == "Desconocido":
            for lugar in lugares:
                if "Línea" in lugar[0]:
                    output_data["transporte"]["total"] += 1
                    output_data["transporte"]["data"].append(lugar)
        elif categoria in ["theatre", "cinema", "library", "community_centre", "arts_centre"]:
            for lugar in lugares:
                output_data["sitios_interes"]["total"] += 1
                output_data["sitios_interes"]["data"].append(lugar)
        elif categoria in ["townhall", "courthouse"]:
            for lugar in lugares:
                output_data["edificios_administrativos"]["total"] += 1
                output_data["edificios_administrativos"]["data"].append(lugar)
        elif categoria in ["college", "kindergarten", "school", "university"]:
            for lugar in lugares:
                output_data["instituciones_educativas"]["total"] += 1
                output_data["instituciones_educativas"]["data"].append(lugar)
        elif categoria == "clinic":
            for lugar in lugares:
                output_data["centros_salud"]["total"] += 1
                output_data["centros_salud"]["data"].append(lugar)
        elif categoria in ["restaurant", "cafe", "ice_cream", "pub", "fast_food", "bar"]:
            for lugar in lugares:
                output_data["restaurantes"]["total"] += 1
                output_data["restaurantes"]["data"].append(lugar)

    return output_data

# Rent Result
def get_rent_result(m2, room, bedroom, antiquity, neighborhood, lambda_client, function_name):
    payload = {
        "total_area": m2,
        "rooms": room,
        "bedrooms": bedroom,
        "antiquity": antiquity,
        "neighborhood": neighborhood,
        "bathrooms": 0,
        "garages": 0,
    }

    response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload)
    )

    response_json = json.loads(response["Payload"].read())
    body_json = json.loads(response_json["body"])
    return body_json["prediction"][0]

def main_execution(input_data):

    OUTPUT_DATA_JSON = {
        "rent_result_min": None,
        "rent_result_max": None,
        "total_properties": None,
        "total_properties_neighborhood": None,
        "average_price_neighborhood": None,
        "min_price_neighborhood": None,
        "max_price_neighborhood": None,
        "total_properties_amb_neighborhood": None,
        "average_price_amb_neighborhood": None,
        "min_price_amb_neighborhood": None,
        "max_price_amb_neighborhood": None,
        "new_properties_since_last_report_neighborhood": None,
        "removed_properties_since_last_report_neighborhood": None,
        "new_properties_since_last_report": None,
        "removed_properties_since_last_report": None,
        "nearby_places_data": None,
        "price_by_m2_evolution": None,
        "bar_price_by_amb": None,
        "bar_m2_price_by_amb": None,
        "bar_price_by_amb_neighborhood": None,
        "bar_m2_price_by_amb_neighborhood": None,
        "pie_property_amb_distribution": None,
        "bar_price_by_amb_neighborhood": None,
    }

    print("Obteniendo información de S3...")
    load_dotenv()
    s3_client = boto3.client('s3')
    BUCKET_NAME = os.getenv('BUCKET_NAME')

    # Agarramos de zonaprop los dos archivos mas nuevos
    files = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
    sub_files = list()
    for file in files["Contents"]:
        if ("/ZonaProp/STG" in file["Key"]) and (".csv" in file["Key"]):
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

    print("Limpiando información...")
    df_old = clean_data(df_old)
    df_new = clean_data(df_new)

    print("Normalizando información de Barrios...")
    # Aplicar la función de mapeo
    df_old['normalized_neighborhood'] = df_old['neighborhood'].apply(map_neighborhood)
    df_new['normalized_neighborhood'] = df_new['neighborhood'].apply(map_neighborhood)

    # Dropeamos los que tienen valor OTRO
    df_old = df_old[df_old['normalized_neighborhood'] != 'OTRO']
    df_new = df_new[df_new['normalized_neighborhood'] != 'OTRO']

    print("Obteniendo información estadística numérica...")
    OUTPUT_DATA_JSON["total_properties"] = len(df_new)
    OUTPUT_DATA_JSON["total_properties_neighborhood"] = len(df_new[df_new["normalized_neighborhood"] == input_data["neighborhood"]])
    OUTPUT_DATA_JSON["average_price_neighborhood"] = str(df_new[df_new["normalized_neighborhood"] == input_data["neighborhood"]]["price"].mean().astype(int))
    OUTPUT_DATA_JSON["min_price_neighborhood"] = str(df_new[df_new["normalized_neighborhood"] == input_data["neighborhood"]]["price"].min().astype(int))
    OUTPUT_DATA_JSON["max_price_neighborhood"] = str(df_new[df_new["normalized_neighborhood"] == input_data["neighborhood"]]["price"].max().astype(int))
    OUTPUT_DATA_JSON["total_properties_amb_neighborhood"] = len(df_new[(df_new["normalized_neighborhood"] == input_data["neighborhood"]) & (df_new["rooms"] == input_data["rooms"])])
    OUTPUT_DATA_JSON["average_price_amb_neighborhood"] = str(df_new[(df_new["normalized_neighborhood"] == input_data["neighborhood"]) & (df_new["rooms"] == input_data["rooms"])]["price"].mean().astype(int))
    OUTPUT_DATA_JSON["min_price_amb_neighborhood"] = str(df_new[(df_new["normalized_neighborhood"] == input_data["neighborhood"]) & (df_new["rooms"] == input_data["rooms"])]["price"].min().astype(int))
    OUTPUT_DATA_JSON["max_price_amb_neighborhood"] = str(df_new[(df_new["normalized_neighborhood"] == input_data["neighborhood"]) & (df_new["rooms"] == input_data["rooms"])]["price"].max().astype(int))
    OUTPUT_DATA_JSON["new_properties_since_last_report_neighborhood"], OUTPUT_DATA_JSON["removed_properties_since_last_report_neighborhood"] = calculate_new_and_removed_properties_neighborhood(df_old, df_new, input_data["neighborhood"])
    OUTPUT_DATA_JSON["new_properties_since_last_report"], OUTPUT_DATA_JSON["removed_properties_since_last_report"] = calculate_new_and_removed_properties_neighborhood(df_old, df_new)

    print("Obteniendo información de lugares cercanos...")
    lugares_cercanos = obtener_lugares_cercanos("{} ,{} ,CABA , Argentina".format(input_data["street"], input_data["neighborhood"]))
    results = procesar_resultados(lugares_cercanos)
    OUTPUT_DATA_JSON["nearby_places_data"] = results

    print("Obteniendo información de precio de alquiler...")
    min_m2, max_m2 = input_data["total_area"].split("-")
    min_m2 = int(min_m2)
    max_m2 = int(max_m2)
    lambda_client = boto3.client('lambda')

    OUTPUT_DATA_JSON["rent_result_min"] = get_rent_result(min_m2, input_data["rooms"], input_data["bedrooms"], input_data["antiquity"], input_data["neighborhood"], lambda_client, os.getenv('INFERENCE_FUNCTION'))
    OUTPUT_DATA_JSON["rent_result_max"] = get_rent_result(max_m2, input_data["rooms"], input_data["bedrooms"], input_data["antiquity"], input_data["neighborhood"], lambda_client, os.getenv('INFERENCE_FUNCTION'))

    folder_name = f"reporting/reports/{uuid.uuid4()}/"
    input_json_key = folder_name + "input_data.json"
    output_json_key = folder_name + "output_data.json"

    upload_json_to_s3(BUCKET_NAME, input_json_key, input_data)
    upload_json_to_s3(BUCKET_NAME, output_json_key, OUTPUT_DATA_JSON)

    print("Obteniendo información estadística gráfica...")
    price_by_m2_evolution = generate_price_evolution_graph(BUCKET_NAME, folder_name, input_data)
    bar_price_by_amb, bar_m2_price_by_amb = generate_bar_charts(df_new, BUCKET_NAME, folder_name)
    bar_price_by_amb_neighborhood, bar_m2_price_by_amb_neighborhood = generate_bar_charts_neighborhood(df_new, input_data["neighborhood"], BUCKET_NAME, folder_name)
    pie_property_amb_distribution = generate_pie_charts(df_new, BUCKET_NAME, folder_name)
    pie_property_amb_distribution_neighborhood = generate_pie_charts_neighborhood(df_new, input_data["neighborhood"], BUCKET_NAME, folder_name)

    OUTPUT_DATA_JSON["price_by_m2_evolution"] = price_by_m2_evolution
    OUTPUT_DATA_JSON["bar_price_by_amb"] = bar_price_by_amb
    OUTPUT_DATA_JSON["bar_m2_price_by_amb"] = bar_m2_price_by_amb
    OUTPUT_DATA_JSON["bar_price_by_amb_neighborhood"] = bar_price_by_amb_neighborhood
    OUTPUT_DATA_JSON["bar_m2_price_by_amb_neighborhood"] = bar_m2_price_by_amb_neighborhood
    OUTPUT_DATA_JSON["pie_property_amb_distribution"] = pie_property_amb_distribution
    OUTPUT_DATA_JSON["pie_property_amb_distribution_neighborhood"] = pie_property_amb_distribution_neighborhood

    return OUTPUT_DATA_JSON

def lambda_handler(event, context):
    print(f"Received event: {json.dumps(event)}")  # Log the entire event

    # Assume POST if httpMethod is not present (for local testing)
    http_method = event.get("httpMethod", "POST")
    try:
        if http_method == "POST":
            # Handle body content
            if "body" in event and event["body"]:
                try:
                    body = json.loads(event["body"])  # Parse the body
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON format in body.")
            else:
                body = event  # Assume the event itself is the payload when body is absent
            print(f"Parsed body: {body}")  # Log the parsed body
            result = main_execution(body)
            return {
                "statusCode": 200,
                "body": json.dumps({"result": result}),
                "headers": {
                    "Content-Type": "application/json"
                }
            }
        else:
            return {
                "statusCode": 405,
                "body": json.dumps({"error": "Method not allowed. Use POST."}),
                "headers": {
                    "Content-Type": "application/json"
                }
            }
    except ValueError as ve:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(ve)}),
            "headers": {
                "Content-Type": "application/json"
            }
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
            "headers": {
                "Content-Type": "application/json"
            }
        }

