import requests
from geopy.geocoders import Nominatim

# Las categorias disponibles son las siguientes:
"""
Desconocido
Pharmacy
Fast_food
Theatre
Place_of_worship
Restaurant
Parking
Cinema
Bank
Cafe
Parking_entrance
Ice_cream
Library
Fountain
Post_box
Bar
Car_rental
Post_office
Townhall
College
Pub
Bicycle_rental
Bureau_de_change
Dentist
Nightclub
Bench
Veterinary
Kindergarten
School
Taxi
Motorcycle_parking
Drinking_water
Clock
Waste_basket
Waste_disposal
Atm
Ticket_validator
Clinic
Courthouse
Payment_centre
Toilets
Vending_machine
University
Police
"""

# Las que nos importan son
"""
Desconocido -> Buscamos "Linea" y serían paradas de colectivo cercanas

Theatre -> Sitios de Interes
Cinema -> Sitios de Interes
Ice_cream -> Sitios de Interes
Library -> Sitios de Interes
Community_centre -> Sitios de Interes
Arts_centre -> Sitios de Interes

Townhall -> Edificios Administrativos
Courthouse -> Edificios Administrativos

College -> Instituciones Educativas
Kindergarten -> Instituciones Educativas
School -> Instituciones Educativas
University -> Instituciones Educativas

Clinic -> Centros de Salud

Restaurant -> Restaurantes, Cafeterías y Bares
Cafe -> Restaurantes, Cafeterías y Bares
Ice_cream -> Restaurantes, Cafeterías y Bares
Pub -> Restaurantes, Cafeterías y Bares
Fast_food -> Restaurantes, Cafeterías y Bares
Bar -> Restaurantes, Cafeterías y Bares
"""

INPUT_DATA = {
  "total_area": 40,
  "rooms": 2,
  "bedrooms": 1,
  "bathrooms": 1,
  "garages": 1,
  "antiquity": 50,
  "neighborhood": "RECOLETA",
  "street": "Posadas 1725",
}

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

direccion = "{} ,{} ,CABA , Argentina".format(INPUT_DATA["street"], INPUT_DATA["neighborhood"])
lugares_cercanos = obtener_lugares_cercanos(direccion)

results = procesar_resultados(lugares_cercanos)

for categoria, data in results.items():
    print(f"{categoria}: {data['total']}")
    for lugar in data["data"]:
        print(f"\t- {lugar[0]}: {lugar[1]}, {lugar[2]}) ")