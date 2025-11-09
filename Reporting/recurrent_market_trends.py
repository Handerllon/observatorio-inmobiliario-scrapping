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

# Configuración
TRENDS_FOLDER = "reporting/trends"

# Configuración para guardar archivos localmente o en S3
SAVE_LOCAL = False  # Cambiar a True para guardar localmente
LOCAL_SAVE_PATH = "./report_output"  # Ruta local donde se guardarán las métricas

now = datetime.now()
formatted_date = now.strftime("%m_%Y")

FINAL_TRENDS_FOLDER = f"{TRENDS_FOLDER}/{formatted_date}"

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

def save_json(data, key):
    """Guarda JSON localmente o en S3 según configuración"""
    json_string = json.dumps(data, ensure_ascii=False, indent=2)
    
    if SAVE_LOCAL:
        # Guardar localmente
        local_file_path = os.path.join(LOCAL_SAVE_PATH, key)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        with open(local_file_path, 'w', encoding='utf-8') as f:
            f.write(json_string)
        print(f"JSON guardado localmente en {local_file_path}")
    else:
        # Guardar en S3
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=json_string,
            ContentType='application/json'
        )
        print(f"JSON subido exitosamente a s3://{BUCKET_NAME}/{key}")

# ============================================================
# MÉTRICAS PARA PROPIETARIOS
# ============================================================

def generate_owner_trends(df_old, df_new, neighborhood):
    """Genera 5 métricas de tendencias para propietarios"""
    trends = []
    
    df_old_neigh = df_old[df_old['normalized_neighborhood'] == neighborhood]
    df_new_neigh = df_new[df_new['normalized_neighborhood'] == neighborhood]
    
    if df_new_neigh.empty:
        return []
    
    # 1. Tendencia de Precios en el Barrio
    avg_price_old = df_old_neigh['price'].mean() if not df_old_neigh.empty else 0
    avg_price_new = df_new_neigh['price'].mean()
    price_change = ((avg_price_new - avg_price_old) / avg_price_old * 100) if avg_price_old > 0 else 0
    
    trend_direction = "aumentaron" if price_change > 0 else "disminuyeron"
    trend_emoji = "📈" if price_change > 0 else "📉"
    
    trends.append({
        "title": f"{trend_emoji} Evolución de Precios en {neighborhood}",
        "description": f"Los precios promedio de alquiler en {neighborhood} {trend_direction} un {abs(price_change):.1f}% desde el último reporte. El precio promedio actual es de ${avg_price_new:,.0f}, comparado con ${avg_price_old:,.0f} anteriormente. Esta tendencia indica {'una revalorización' if price_change > 0 else 'una oportunidad de ajuste'} en la zona.",
        "value": f"{price_change:+.1f}%",
        "metric_type": "price_trend"
    })
    
    # 2. Nivel de Competencia en la Zona
    total_properties = len(df_new_neigh)
    avg_rooms = df_new_neigh['rooms'].mode()[0] if not df_new_neigh.empty else 2
    similar_properties = len(df_new_neigh[df_new_neigh['rooms'] == avg_rooms])
    competition_rate = (similar_properties / total_properties * 100) if total_properties > 0 else 0
    
    competition_level = "alta" if competition_rate > 40 else "moderada" if competition_rate > 25 else "baja"
    
    trends.append({
        "title": f"🏘️ Competencia en el Mercado de {neighborhood}",
        "description": f"Actualmente hay {total_properties} propiedades disponibles en {neighborhood}. Las propiedades de {int(avg_rooms)} ambientes representan el {competition_rate:.1f}% del mercado ({similar_properties} unidades), indicando una competencia {competition_level}. {'Destacar características únicas de tu propiedad será clave para diferenciarte.' if competition_level == 'alta' else 'Existe una buena oportunidad para posicionar tu propiedad en el mercado.'}",
        "value": f"{total_properties} propiedades",
        "metric_type": "competition"
    })
    
    # 3. Precio por m² Comparativo
    price_per_m2_neigh = (df_new_neigh['price'] / df_new_neigh['total_area']).mean()
    price_per_m2_global = (df_new['price'] / df_new['total_area']).mean()
    m2_diff = ((price_per_m2_neigh - price_per_m2_global) / price_per_m2_global * 100)
    
    position = "por encima" if m2_diff > 0 else "por debajo"
    
    trends.append({
        "title": f"💰 Valor por m² en {neighborhood}",
        "description": f"El precio promedio por m² en {neighborhood} es de ${price_per_m2_neigh:,.0f}, lo que está {abs(m2_diff):.1f}% {position} del promedio general de CABA (${price_per_m2_global:,.0f}/m²). Esto {'refleja la valorización de la zona' if m2_diff > 0 else 'representa una oportunidad competitiva'} y {'justifica precios premium por ubicación y servicios' if m2_diff > 5 else 'sugiere ajustar la estrategia de precios según características específicas'}.",
        "value": f"${price_per_m2_neigh:,.0f}/m²",
        "metric_type": "price_per_m2"
    })
    
    # 4. Rotación del Mercado
    old_ids = set(df_old_neigh['zonaprop_code'].unique())
    new_ids = set(df_new_neigh['zonaprop_code'].unique())
    
    new_listings = len(new_ids - old_ids)
    removed_listings = len(old_ids - new_ids)
    rotation_rate = ((new_listings + removed_listings) / len(old_ids) * 100) if len(old_ids) > 0 else 0
    
    market_speed = "rápida" if rotation_rate > 30 else "moderada" if rotation_rate > 15 else "lenta"
    
    trends.append({
        "title": f"⚡ Velocidad del Mercado en {neighborhood}",
        "description": f"La rotación del mercado en {neighborhood} es {market_speed}, con {new_listings} propiedades nuevas y {removed_listings} propiedades retiradas desde el último reporte. Una rotación del {rotation_rate:.1f}% indica {'alta demanda y movimiento constante' if rotation_rate > 30 else 'un mercado estable con oportunidades' if rotation_rate > 15 else 'un mercado selectivo donde la paciencia es clave'}. {'Las propiedades bien presentadas se alquilan rápidamente.' if rotation_rate > 30 else 'Tómate tiempo para optimizar tu estrategia de marketing.' if rotation_rate < 15 else 'Un pricing competitivo puede acelerar el proceso.'}",
        "value": f"{rotation_rate:.1f}% rotación",
        "metric_type": "market_rotation"
    })
    
    # 5. Segmento de Mayor Demanda
    rooms_distribution = df_new_neigh['rooms'].value_counts()
    top_segment = rooms_distribution.index[0] if len(rooms_distribution) > 0 else 2
    top_segment_count = rooms_distribution.iloc[0] if len(rooms_distribution) > 0 else 0
    top_segment_pct = (top_segment_count / len(df_new_neigh) * 100) if len(df_new_neigh) > 0 else 0
    
    avg_price_segment = df_new_neigh[df_new_neigh['rooms'] == top_segment]['price'].mean()
    
    trends.append({
        "title": f"🎯 Segmento Más Demandado en {neighborhood}",
        "description": f"Las propiedades de {int(top_segment)} ambientes dominan el mercado en {neighborhood}, representando el {top_segment_pct:.1f}% de la oferta ({int(top_segment_count)} unidades). El precio promedio para este segmento es de ${avg_price_segment:,.0f}. {'Si tu propiedad coincide con este perfil, estás en el segmento con mayor visibilidad.' if top_segment_pct > 35 else 'Aunque no es el segmento dominante, existen nichos específicos con menos competencia.'}",
        "value": f"{int(top_segment)} ambientes",
        "metric_type": "top_segment"
    })
    
    return trends

# ============================================================
# MÉTRICAS PARA AGENTES INMOBILIARIOS
# ============================================================

def generate_agent_trends(df_old, df_new, neighborhood):
    """Genera 5 métricas de tendencias para agentes inmobiliarios"""
    trends = []
    
    df_old_neigh = df_old[df_old['normalized_neighborhood'] == neighborhood]
    df_new_neigh = df_new[df_new['normalized_neighborhood'] == neighborhood]
    
    if df_new_neigh.empty:
        return []
    
    # 1. Oportunidades de Crecimiento en la Zona
    growth_rate = ((len(df_new_neigh) - len(df_old_neigh)) / len(df_old_neigh) * 100) if len(df_old_neigh) > 0 else 0
    
    growth_status = "creciendo" if growth_rate > 5 else "estable" if growth_rate > -5 else "contrayéndose"
    opportunity_level = "Alta" if abs(growth_rate) > 10 else "Moderada" if abs(growth_rate) > 5 else "Estable"
    
    trends.append({
        "title": f"📊 Dinámica del Mercado en {neighborhood}",
        "description": f"El mercado de alquileres en {neighborhood} está {growth_status} con un cambio del {growth_rate:+.1f}% en la oferta de propiedades. {'Esto indica una zona con alta actividad y oportunidades para captar nuevos clientes propietarios.' if growth_rate > 5 else 'La estabilidad del mercado permite construir relaciones duraderas con clientes.' if abs(growth_rate) <= 5 else 'La contracción puede indicar propiedades alquiladas rápidamente o retiradas del mercado.'} Nivel de oportunidad: {opportunity_level}.",
        "value": f"{growth_rate:+.1f}%",
        "metric_type": "market_growth"
    })
    
    # 2. Análisis de Precios por Segmento
    price_by_rooms = df_new_neigh.groupby('rooms')['price'].agg(['mean', 'count']).reset_index()
    price_by_rooms = price_by_rooms.sort_values('count', ascending=False)
    
    if len(price_by_rooms) > 0:
        top_room = price_by_rooms.iloc[0]['rooms']
        top_price = price_by_rooms.iloc[0]['mean']
        top_count = price_by_rooms.iloc[0]['count']
        
        if len(price_by_rooms) > 1:
            second_room = price_by_rooms.iloc[1]['rooms']
            second_price = price_by_rooms.iloc[1]['mean']
            price_gap = ((top_price - second_price) / second_price * 100)
        else:
            price_gap = 0
    else:
        top_room, top_price, top_count, price_gap = 2, 0, 0, 0
    
    trends.append({
        "title": f"💼 Segmentación de Precios en {neighborhood}",
        "description": f"El segmento de {int(top_room)} ambientes lidera el mercado con {int(top_count)} propiedades y un precio promedio de ${top_price:,.0f}. {'Existe una diferencia de precio significativa entre segmentos, lo que permite estrategias de posicionamiento diferenciadas.' if abs(price_gap) > 15 else 'Los precios entre segmentos son relativamente uniformes, facilitando comparaciones directas.'} Identificar el segmento objetivo de cada cliente es clave para maximizar conversiones.",
        "value": f"${top_price:,.0f}",
        "metric_type": "price_segmentation"
    })
    
    # 3. Comparativa con Barrios Cercanos
    # Calcular promedio de barrios similares (top 3 barrios con más propiedades excluyendo el actual)
    other_neighborhoods = df_new[df_new['normalized_neighborhood'] != neighborhood]
    top_neighborhoods = other_neighborhoods.groupby('normalized_neighborhood')['price'].agg(['mean', 'count']).reset_index()
    top_neighborhoods = top_neighborhoods.sort_values('count', ascending=False).head(3)
    
    avg_price_neigh = df_new_neigh['price'].mean()
    avg_price_competitors = top_neighborhoods['mean'].mean() if len(top_neighborhoods) > 0 else avg_price_neigh
    
    competitive_position = ((avg_price_neigh - avg_price_competitors) / avg_price_competitors * 100)
    position_desc = "más económico" if competitive_position < -5 else "similar" if abs(competitive_position) <= 5 else "más premium"
    
    trends.append({
        "title": f"🗺️ Posición Competitiva de {neighborhood}",
        "description": f"{neighborhood} se posiciona como un barrio {position_desc} comparado con zonas competidoras (diferencia: {competitive_position:+.1f}%). Con un precio promedio de ${avg_price_neigh:,.0f} vs ${avg_price_competitors:,.0f} en barrios similares, {'puedes destacar el valor competitivo y ubicación estratégica' if competitive_position < 0 else 'debes enfatizar las ventajas exclusivas y calidad de vida' if competitive_position > 5 else 'tienes flexibilidad para argumentar tanto precio como valor agregado'}.",
        "value": f"{competitive_position:+.1f}%",
        "metric_type": "competitive_position"
    })
    
    # 4. Inventario Disponible por Tipología
    rooms_dist = df_new_neigh['rooms'].value_counts().to_dict()
    total_properties = len(df_new_neigh)
    
    inventory_summary = ", ".join([f"{int(k)} amb: {v}" for k, v in sorted(rooms_dist.items())[:3]])
    
    # Identificar segmento con menos competencia
    if len(rooms_dist) > 0:
        min_competition_segment = min(rooms_dist, key=rooms_dist.get)
        min_competition_count = rooms_dist[min_competition_segment]
        min_competition_pct = (min_competition_count / total_properties * 100)
    else:
        min_competition_segment = 2
        min_competition_pct = 0
    
    trends.append({
        "title": f"📦 Disponibilidad de Inventario en {neighborhood}",
        "description": f"Actualmente hay {total_properties} propiedades activas en {neighborhood}. Distribución: {inventory_summary}. El segmento de {int(min_competition_segment)} ambientes tiene menor competencia ({min_competition_pct:.1f}% del mercado), {'representando una oportunidad para enfocarse en este nicho' if min_competition_pct < 20 else 'aunque todos los segmentos muestran actividad'}. Mantén un pipeline diversificado para atender diferentes perfiles de inquilinos.",
        "value": f"{total_properties} unidades",
        "metric_type": "inventory"
    })
    
    # 5. Tendencia de Precio por m²
    df_old_neigh['price_per_m2'] = df_old_neigh['price'] / df_old_neigh['total_area']
    df_new_neigh['price_per_m2'] = df_new_neigh['price'] / df_new_neigh['total_area']
    
    old_price_m2 = df_old_neigh['price_per_m2'].mean() if not df_old_neigh.empty else 0
    new_price_m2 = df_new_neigh['price_per_m2'].mean()
    m2_trend = ((new_price_m2 - old_price_m2) / old_price_m2 * 100) if old_price_m2 > 0 else 0
    
    # Encontrar propiedades con mejor valor por m²
    best_value_properties = df_new_neigh.nsmallest(int(len(df_new_neigh) * 0.25), 'price_per_m2')
    best_value_avg = best_value_properties['price_per_m2'].mean()
    
    trends.append({
        "title": f"📏 Análisis de Valor por m² en {neighborhood}",
        "description": f"El precio por m² en {neighborhood} {'aumentó' if m2_trend > 0 else 'disminuyó'} un {abs(m2_trend):.1f}%, alcanzando ${new_price_m2:,.0f}/m². {'Esta apreciación indica valorización de la zona' if m2_trend > 2 else 'La estabilidad facilita la negociación' if abs(m2_trend) <= 2 else 'Esto presenta oportunidades para propietarios flexibles'}. Las propiedades con mejor relación precio/m² promedian ${best_value_avg:,.0f}/m², ideales para inquilinos conscientes del presupuesto.",
        "value": f"${new_price_m2:,.0f}/m²",
        "metric_type": "price_per_m2_trend"
    })
    
    return trends

# ============================================================
# MÉTRICAS PARA INQUILINOS
# ============================================================

def generate_tenant_trends(df_old, df_new, neighborhood):
    """Genera 5 métricas de tendencias para inquilinos"""
    trends = []
    
    df_old_neigh = df_old[df_old['normalized_neighborhood'] == neighborhood]
    df_new_neigh = df_new[df_new['normalized_neighborhood'] == neighborhood]
    
    if df_new_neigh.empty:
        return []
    
    # 1. Mejor Momento para Alquilar
    avg_price_old = df_old_neigh['price'].mean() if not df_old_neigh.empty else 0
    avg_price_new = df_new_neigh['price'].mean()
    price_change = ((avg_price_new - avg_price_old) / avg_price_old * 100) if avg_price_old > 0 else 0
    
    timing_status = "favorable" if price_change < 0 else "menos favorable" if price_change > 5 else "neutral"
    action_recommendation = "Es un buen momento para buscar y negociar." if price_change < 0 else "Los precios están subiendo, considera asegurar pronto." if price_change > 5 else "El mercado está estable, negocia según tu presupuesto."
    
    trends.append({
        "title": f"⏰ Momento de Mercado en {neighborhood}",
        "description": f"El momento actual para alquilar en {neighborhood} es {timing_status}. Los precios {'bajaron' if price_change < 0 else 'subieron'} un {abs(price_change):.1f}% desde el último reporte (${avg_price_old:,.0f} → ${avg_price_new:,.0f}). {action_recommendation} {'Aprovecha las oportunidades disponibles antes de que cambien las condiciones.' if timing_status == 'favorable' else 'Mantente atento a nuevas publicaciones.' if timing_status == 'menos favorable' else 'Compara múltiples opciones para encontrar la mejor relación calidad-precio.'}",
        "value": f"{price_change:+.1f}%",
        "metric_type": "timing"
    })
    
    # 2. Disponibilidad de Opciones
    total_available = len(df_new_neigh)
    new_listings = len(set(df_new_neigh['zonaprop_code']) - set(df_old_neigh['zonaprop_code']))
    availability_status = "alta" if total_available > 50 else "moderada" if total_available > 20 else "limitada"
    
    trends.append({
        "title": f"🏠 Disponibilidad en {neighborhood}",
        "description": f"Actualmente hay {total_available} propiedades disponibles en {neighborhood}, con {new_listings} publicaciones nuevas desde el último reporte. La disponibilidad es {availability_status}, lo que {'te da múltiples opciones para comparar y elegir' if availability_status == 'alta' else 'requiere que actúes rápido en propiedades que te interesen' if availability_status == 'limitada' else 'permite una búsqueda selectiva'}. {'Aprovecha la variedad para negociar mejores condiciones.' if total_available > 50 else 'Programa visitas cuanto antes para no perder oportunidades.' if total_available < 20 else 'Toma tu tiempo pero mantente atento a nuevas publicaciones.'}",
        "value": f"{total_available} propiedades",
        "metric_type": "availability"
    })
    
    # 3. Mejores Oportunidades de Precio
    df_new_neigh['price_per_m2'] = df_new_neigh['price'] / df_new_neigh['total_area']
    
    # Encontrar el percentil 25 (mejores precios)
    price_threshold = df_new_neigh['price'].quantile(0.25)
    best_deals = df_new_neigh[df_new_neigh['price'] <= price_threshold]
    best_deals_count = len(best_deals)
    best_deals_avg = best_deals['price'].mean() if not best_deals.empty else 0
    best_deals_m2 = best_deals['price_per_m2'].mean() if not best_deals.empty else 0
    
    savings_pct = ((avg_price_new - best_deals_avg) / avg_price_new * 100) if best_deals_avg > 0 else 0
    
    trends.append({
        "title": f"💸 Mejores Oportunidades en {neighborhood}",
        "description": f"Identificamos {best_deals_count} propiedades con precios competitivos (hasta ${price_threshold:,.0f}), con un promedio de ${best_deals_avg:,.0f} y ${best_deals_m2:,.0f}/m². Estas oportunidades representan un ahorro de hasta {savings_pct:.1f}% respecto al precio promedio de la zona (${avg_price_new:,.0f}). {'Estas propiedades suelen alquilarse rápido' if best_deals_count < 10 else 'Tienes buenas opciones para comparar'}, así que {'agenda visitas inmediatamente' if best_deals_count < 10 else 'evalúa características adicionales como ubicación y estado'}.",
        "value": f"${best_deals_avg:,.0f}",
        "metric_type": "best_deals"
    })
    
    # 4. Características de Propiedades Disponibles
    # Análisis de distribución por tamaño
    size_ranges = [
        (0, 30, "Compactas (hasta 30m²)"),
        (30, 50, "Medianas (30-50m²)"),
        (50, 80, "Amplias (50-80m²)"),
        (80, float('inf'), "Muy amplias (+80m²)")
    ]
    
    size_distribution = []
    for min_size, max_size, label in size_ranges:
        props = df_new_neigh[(df_new_neigh['total_area'] >= min_size) & (df_new_neigh['total_area'] < max_size)]
        count = len(props)
        if count > 0:
            pct = (count / total_available * 100)
            avg_rooms = props['rooms'].mean()
            avg_bathrooms = props['bathrooms'].mean()
            size_distribution.append({
                'label': label,
                'count': count,
                'pct': pct,
                'avg_rooms': avg_rooms,
                'avg_bathrooms': avg_bathrooms
            })
    
    # Encontrar el segmento más disponible
    if size_distribution:
        top_size = max(size_distribution, key=lambda x: x['count'])
        size_summary = f"{top_size['label']}: {int(top_size['count'])} propiedades ({top_size['pct']:.0f}%), promedio {top_size['avg_rooms']:.1f} ambientes y {top_size['avg_bathrooms']:.1f} baños"
        
        # Contar propiedades con características premium (más de 1 baño, cochera, etc)
        premium_features = len(df_new_neigh[(df_new_neigh['bathrooms'] > 1) | (df_new_neigh['garages'] > 0)])
        premium_pct = (premium_features / total_available * 100) if total_available > 0 else 0
    else:
        size_summary = "No hay datos suficientes"
        premium_pct = 0
    
    trends.append({
        "title": f"🏗️ Perfil de Propiedades en {neighborhood}",
        "description": f"El tipo de propiedad más común en {neighborhood} son las {top_size['label'] if size_distribution else 'N/A'} con {int(top_size['count']) if size_distribution else 0} unidades disponibles. {size_summary}. Además, el {premium_pct:.0f}% de las propiedades cuentan con características premium como más de un baño o cochera. {'Hay buena variedad de opciones para diferentes necesidades de espacio' if len(size_distribution) >= 3 else 'La oferta está concentrada en un tipo específico de propiedad'}.",
        "value": f"{int(top_size['count']) if size_distribution else 0} en segmento principal",
        "metric_type": "property_profile"
    })
    
    # 5. Velocidad del Mercado (para inquilinos)
    old_ids = set(df_old_neigh['zonaprop_code'].unique())
    new_ids = set(df_new_neigh['zonaprop_code'].unique())
    
    removed_listings = len(old_ids - new_ids)
    rotation_rate = (removed_listings / len(old_ids) * 100) if len(old_ids) > 0 else 0
    
    market_speed = "muy dinámico" if rotation_rate > 40 else "dinámico" if rotation_rate > 25 else "pausado"
    urgency = "alta" if rotation_rate > 40 else "moderada" if rotation_rate > 25 else "baja"
    
    trends.append({
        "title": f"⚡ Ritmo del Mercado en {neighborhood}",
        "description": f"El mercado en {neighborhood} es {market_speed}, con {removed_listings} propiedades alquiladas o retiradas desde el último reporte (rotación del {rotation_rate:.1f}%). {'Las propiedades se alquilan rápidamente, por lo que debes actuar con agilidad' if rotation_rate > 40 else 'Hay movimiento constante, mantente atento a nuevas publicaciones' if rotation_rate > 25 else 'Tienes tiempo para evaluar opciones con calma'}. Urgencia recomendada: {urgency}. {'Prepara tu documentación con anticipación para cerrar rápido.' if urgency == 'alta' else 'Visita varias opciones antes de decidir.' if urgency == 'baja' else 'Equilibra análisis con prontitud en la decisión.'}",
        "value": f"{rotation_rate:.1f}% rotación",
        "metric_type": "market_speed"
    })
    
    return trends

# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

print("Obteniendo información de S3...")
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

print(f"Archivo actual: {newest_file}")
print(f"Archivo anterior: {previous_file}")

response = s3_client.get_object(Bucket=BUCKET_NAME, Key=newest_file)
csv_data = response['Body'].read().decode('utf-8')
df_new = pd.read_csv(StringIO(csv_data))

response = s3_client.get_object(Bucket=BUCKET_NAME, Key=previous_file)
csv_data = response['Body'].read().decode('utf-8')
df_old = pd.read_csv(StringIO(csv_data))

print("Limpiando datos...")
df_old = clean_data(df_old)
df_new = clean_data(df_new)

print("Normalizando barrios...")
# Aplicar la función de mapeo
df_old['normalized_neighborhood'] = df_old['neighborhood'].apply(map_neighborhood)
df_new['normalized_neighborhood'] = df_new['neighborhood'].apply(map_neighborhood)

# Dropeamos los que tienen valor OTRO
df_old = df_old[df_old['normalized_neighborhood'] != 'OTRO']
df_new = df_new[df_new['normalized_neighborhood'] != 'OTRO']

# Borramos todos los contenidos dentro de la carpeta (solo si no es guardado local)
if not SAVE_LOCAL:
    print(f"Borrando contenido de la carpeta {FINAL_TRENDS_FOLDER}")
    objects_to_delete = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=FINAL_TRENDS_FOLDER+"/")
    if objects_to_delete["KeyCount"] > 0:
        s3_client.delete_objects(Bucket=BUCKET_NAME, Delete={'Objects': [{'Key': obj['Key']} for obj in objects_to_delete['Contents']]})
    print(f"Contenido de la carpeta {FINAL_TRENDS_FOLDER} borrado")

# Generamos las métricas para cada barrio
for neighborhood in valid_neighborhoods:
    print(f"\n{'='*60}")
    print(f"Generando tendencias de mercado para {neighborhood}")
    print(f"{'='*60}")
    
    # Generar tendencias para cada perfil de usuario
    owner_trends = generate_owner_trends(df_old, df_new, neighborhood)
    agent_trends = generate_agent_trends(df_old, df_new, neighborhood)
    tenant_trends = generate_tenant_trends(df_old, df_new, neighborhood)
    
    # Consolidar todas las tendencias
    all_trends = {
        "neighborhood": neighborhood,
        "date": formatted_date,
        "generated_at": datetime.now().isoformat(),
        "owner_trends": owner_trends,
        "agent_trends": agent_trends,
        "tenant_trends": tenant_trends
    }
    
    # Guardar JSON
    json_key = f"{FINAL_TRENDS_FOLDER}/{neighborhood}/market_trends.json"
    save_json(all_trends, json_key)
    
    print(f"✅ Tendencias generadas para {neighborhood}")
    print(f"   - Propietarios: {len(owner_trends)} métricas")
    print(f"   - Agentes: {len(agent_trends)} métricas")
    print(f"   - Inquilinos: {len(tenant_trends)} métricas")

print(f"\n{'='*60}")
print("✅ Proceso completado exitosamente!")
print(f"{'='*60}")

