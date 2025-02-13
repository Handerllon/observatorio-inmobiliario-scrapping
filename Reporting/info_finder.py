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

Townhall -> Edificios Administrativos
Courthouse -> Edificios Administrativos

College -> Instituciones Educativas
Kindergarten -> Instituciones Educativas
School -> Instituciones Educativas
University -> Instituciones Educativas

Clinic -> Centros de Salud


"""

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
    print(coordenadas)
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

direccion = "Posadas 1725, CABA, Argentina"
lugares_cercanos = obtener_lugares_cercanos(direccion)

for categoria, lugares in lugares_cercanos.items():
    print(f"\n{categoria.capitalize()}:")
    for nombre, lat, lon in lugares:
        print(f" - {nombre} (Lat: {lat}, Lon: {lon})")


for categoria, lugares in lugares_cercanos.items():
    print(f"\n{categoria.capitalize()}:")