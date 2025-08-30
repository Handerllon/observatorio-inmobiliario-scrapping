from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
import undetected_chromedriver as uc
from seleniumbase import Driver
import random
import re
import os
from datetime import datetime

### Custom Imports
from utils import log
from io import StringIO

DOMAIN_URL = "https://www.argenprop.com.ar"
BASE_URL = "https://www.argenprop.com/departamentos/alquiler"

# Lista de barrios de Capital Federal para iterar
BARRIOS = [
    "abasto",
    "almagro", 
    "barracas",
    "br-norte",
    "belgrano",
    "boca",
    "boedo",
    "caballito",
    "centro",
    "chacarita",
    "colegiales",
    "congreso",
    "constitucion",
    "flores",
    "floresta",
    "liniers",
    "mataderos",
    "nuñez",
    "once",
    "palermo",
    "parque-centenario",
    "parque-chacabuco",
    "parque-patricios",
    "paternal",
    "saavedra",
    "san-cristobal",
    "san-telmo"
]

# Lista de User Agents para rotar
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def random_sleep(min_seconds=2, max_seconds=5):
    """Espera aleatoria entre min_seconds y max_seconds"""
    sleep_time = random.uniform(min_seconds, max_seconds)
    sleep(sleep_time)

def gen_driver():
    """Genera un driver de Chrome optimizado para EC2"""
    try:
        # Usar un User Agent aleatorio
        user_agent = get_random_user_agent()
        
        log("INFO", f"Creating driver with User Agent: {user_agent[:50]}...")
            
        # Fallback a undetected-chromedriver con configuración para EC2
        chrome_options = uc.ChromeOptions()
        chrome_options.add_argument('--headless=new')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-web-security')
        chrome_options.add_argument('--disable-features=VizDisplayCompositor,TranslateUI')
        chrome_options.add_argument('--disable-background-timer-throttling')
        chrome_options.add_argument('--disable-backgrounding-occluded-windows')
        chrome_options.add_argument('--disable-renderer-backgrounding')
        chrome_options.add_argument('--memory-pressure-off')
        chrome_options.add_argument(f'--user-agent={user_agent}')
        
        driver = uc.Chrome(options=chrome_options)
        log("INFO", "undetected-chromedriver created successfully as fallback")
        
        # Configurar timeouts para evitar problemas de conexión en EC2
        try:
            driver.set_page_load_timeout(120)  # 120 segundos para cargar página
            driver.implicitly_wait(20)         # 20 segundos para elementos
            driver.set_script_timeout(60)      # 60 segundos para scripts
        except:
            log("WARNING", "Could not set timeouts (fallback driver)")
        
        log("INFO", "Driver created successfully with EC2-optimized configuration")
        return driver
    except Exception as e:
        log("ERROR", f"Error in generating driver: {e}")
        return None

def bs4_parse_raw_html(raw_html):
    """Parsea el HTML usando BeautifulSoup para extraer datos de propiedades"""
    return_list = list()
    
    try:
        # Initialize BeautifulSoup
        soup = BeautifulSoup(raw_html, 'html.parser')

        # We look for the container of the property cards
        postings_container = soup.find('div', class_='listing-container')

        # Process each property card
        if postings_container:
            property_cards = soup.select("div[class^=listing__item]")
            log("INFO", f"Found {len(property_cards)} property cards")
            
            # Process each property card
            for idx, card in enumerate(property_cards):
                try:
                    property_url = "https://www.argenprop.com" + card.find('a', class_='card')['href']
                    currency = card.find('p', class_='card__price').find('span', class_='card__currency').text.strip() if card.find('p', class_='card__price').find('span', class_='card__currency') else None
                    only_price = card.find('p', class_='card__price').find('span', class_='card__currency').next_sibling.strip() if card.find('p', class_='card__price').find('span', class_='card__currency') else None
                    price = str(currency) + " " + str(only_price)
                    expenses = card.find('span', class_='card__expenses').text.strip().replace("+ ", "") if card.find('span', class_='card__expenses') else None
                    address = card.find('p', class_='card__address').text.strip() if card.find('p', class_='card__address') else None
                    location = card.find('p', class_='card__title--primary').text.strip() if card.find('p', class_='card__title--primary') else None
                    features = soup.find('ul', class_='card__main-features')
                    features_text = [li.text.strip() for li in features.find_all('li')] if features else None
                    description = card.find('h2', class_='card__title').text.strip() if card.find('h2', class_='card__title') else None
                    
                    return_list.append({
                        'property_url': property_url,
                        'argenprop_code': property_url.split("-")[-1],
                        'price': price,
                        'expenses': expenses,
                        'address': address,
                        'location': location,
                        'features': features_text,
                        'description': description
                    })
                except Exception as card_error:
                    log("WARNING", f"Error processing property card {idx}: {card_error}")
                    continue
        else:
            log("WARNING", "No property cards container found in HTML")
            
    except Exception as e:
        log("ERROR", f"Error parsing HTML: {e}")
    
    return return_list

def extract_property_ids(raw_html):
    """Extrae los IDs de las propiedades del HTML para detectar duplicados"""
    try:
        soup = BeautifulSoup(raw_html, 'html.parser')
        property_cards = soup.select("div[class^=listing__item]")
        
        property_ids = set()
        for card in property_cards:
            try:
                property_url = card.find('a', class_='card')['href']
                property_id = property_url.split("-")[-1]
                property_ids.add(property_id)
            except:
                continue
                
        return property_ids
    except:
        return set()

def simple_navigation(driver, url):
    """Navegación simple sin referers ni navegación orgánica"""
    try:
        log("INFO", f"Navigating to: {url}")
        driver.get(url)
        
        # Esperar que la página cargue
        random_sleep(3, 6)
        
        # Verificar si la página cargó correctamente
        if "argenprop" in driver.current_url.lower():
            log("INFO", "Page loaded successfully")
            return True
        else:
            log("WARNING", "Page may not have loaded correctly")
            return False
            
    except Exception as e:
        log("ERROR", f"Error in simple navigation: {e}")
        return False

def start_scrapping(out_file, iterations, s3_client, bucket_name):
    """Función principal de scraping simplificada para EC2"""
    driver = gen_driver()
    if driver is None:
        log("ERROR", "Failed to create driver. Aborting scraping process.")
        return
    
    out_values = list()
    failed_pages = []
    consecutive_failures = 0
    max_consecutive_failures = 3

    log("INFO", f"Starting simplified scraping process for ArgenProp - {len(BARRIOS)} barrios, {iterations} pages each")

    try:
        # Iterar por cada barrio, luego por páginas
        for barrio_idx, barrio in enumerate(BARRIOS):
            log("INFO", f"Starting scraping for barrio: {barrio.upper()} ({barrio_idx + 1}/{len(BARRIOS)})")
            
            # Variables para detectar contenido duplicado
            previous_property_ids = set()
            consecutive_duplicates = 0
            
            # Tomar un descanso entre barrios
            if barrio_idx > 0:
                log("INFO", f"Taking break before switching to {barrio}...")
                random_sleep(10, 20)
            
            for page_number in range(1, iterations + 1):
                if page_number == 1:
                    url = f"{BASE_URL}/{barrio}"
                else:
                    url = f"{BASE_URL}/{barrio}?pagina-{page_number}"
                
                log("INFO", f"Scraping {barrio} - page {page_number}/{iterations} - URL: {url}")

                try:
                    # Navegación simple sin referers
                    if not simple_navigation(driver, url):
                        log("WARNING", f"Failed to load {barrio} page {page_number}")
                        failed_pages.append(f"{barrio}-{page_number}")
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            log("ERROR", f"Too many consecutive failures ({consecutive_failures}). Taking longer break...")
                            random_sleep(60, 120)
                            consecutive_failures = 0
                        continue

                    # Resetear contador de fallos consecutivos
                    consecutive_failures = 0

                    # Espera aleatoria entre páginas
                    random_sleep(2, 5)

                    # Verificar si llegamos al final de las páginas para este barrio
                    current_url = driver.current_url
                    
                    # Detectar redirección a una página anterior (indica fin de resultados)
                    if page_number > 1:
                        current_page_match = re.search(r'pagina-(\d+)', current_url)
                        if current_page_match:
                            current_page = int(current_page_match.group(1))
                            if current_page < page_number:
                                log("INFO", f"Reached end of results for {barrio} (redirected to page {current_page} when requesting {page_number}). Moving to next barrio.")
                                break
                        elif 'pagina-' not in current_url and page_number > 1:
                            log("INFO", f"Reached end of results for {barrio} (redirected to page 1 when requesting {page_number}). Moving to next barrio.")
                            break

                    # Extraer HTML y analizar
                    raw_html = driver.page_source

                    # Detectar contenido duplicado comparando IDs de propiedades
                    current_property_ids = extract_property_ids(raw_html)
                    if page_number > 1 and current_property_ids and current_property_ids == previous_property_ids:
                        consecutive_duplicates += 1
                        log("INFO", f"Detected duplicate content on {barrio} page {page_number} (consecutive: {consecutive_duplicates})")
                        
                        if consecutive_duplicates >= 2:
                            log("INFO", f"Multiple consecutive duplicates detected for {barrio}. Moving to next barrio.")
                            break
                    else:
                        consecutive_duplicates = 0
                        previous_property_ids = current_property_ids.copy()

                    # Extraer datos de propiedades
                    try:
                        extracted_values = bs4_parse_raw_html(raw_html)
                        log("INFO", f"Extracted {len(extracted_values)} property cards from {barrio} page {page_number}")
                        
                        if not extracted_values:
                            log("WARNING", f"No property cards found on {barrio} page {page_number}")
                            if page_number > 1:
                                log("INFO", f"No more results for {barrio}. Moving to next barrio.")
                                break
                            else:
                                failed_pages.append(f"{barrio}-{page_number}")
                        else:
                            out_values.extend(extracted_values)
                            
                    except Exception as parse_error:
                        log("ERROR", f"Error parsing page {barrio} page {page_number}: {parse_error}")
                        failed_pages.append(f"{barrio}-{page_number}")

                except Exception as page_error:
                    log("ERROR", f"Error processing {barrio} page {page_number}: {page_error}")
                    failed_pages.append(f"{barrio}-{page_number}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        log("ERROR", f"Too many consecutive failures ({consecutive_failures}). Taking longer break...")
                        random_sleep(60, 120)
                        consecutive_failures = 0

    except Exception as e:
        log("ERROR", f"Critical error in scraping process: {e}")
    
    finally:
        # Cerrar driver
        try:
            driver.quit()
            log("INFO", "Driver closed successfully")
        except:
            log("WARNING", "Error closing driver")

    # Generar reporte final
    log("INFO", f"Scraping completed. Total properties extracted: {len(out_values)}")
    if failed_pages:
        log("WARNING", f"Failed pages: {failed_pages}")
    
    # Guardar resultados
    try:
        if out_values:
            df = pd.DataFrame(out_values)
            
            # Crear directorio de salida si no existe
            output_dir = os.path.dirname(out_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Guardar archivo
            df.to_csv(out_file, index=False)
            log("INFO", f"Results saved to: {out_file}")
            
            # Subir a S3 si está configurado
            if s3_client and bucket_name:
                try:
                    s3_key = f"scrapping/ArgenProp/STOCK/{os.path.basename(out_file)}"
                    s3_client.put_object(
                        Bucket=bucket_name,
                        Key=s3_key,
                        Body=df.to_csv(index=False)
                    )
                    log("INFO", f"Results uploaded to S3: s3://{bucket_name}/{s3_key}")
                except Exception as s3_error:
                    log("ERROR", f"Failed to upload to S3: {s3_error}")
        else:
            log("WARNING", "No data to save")
            
    except Exception as save_error:
        log("ERROR", f"Error saving results: {save_error}")

    return out_values