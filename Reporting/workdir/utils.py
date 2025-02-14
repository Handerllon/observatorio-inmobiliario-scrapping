import pandas as pd
import numpy as np

def clean_data(df):

    df_to_clean = df.copy(deep=True)
    # Hay varias columnas que para la porción de ML no nos interesan. Las vamos a borrar
    try:
        df_to_clean.drop(columns=["location", "created_at", "updated_at", "property_url", "address", "zonaprop_code"]
                , inplace=True)
    except KeyError:
        df_to_clean.drop(columns=["location", "property_url", "address", "zonaprop_code"]
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

    return df_to_clean