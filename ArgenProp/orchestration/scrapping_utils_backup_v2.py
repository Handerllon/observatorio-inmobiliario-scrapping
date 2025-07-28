from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
import undetected_chromedriver as uc
from selenium_stealth import stealth
from seleniumbase import Driver
import random
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
import requests
import time
import json

### Custom Imports
from utils import log
from io import StringIO

DOMAIN_URL = "https://www.argenprop.com.ar"
BASE_URL = "https://www.argenprop.com/departamentos/alquiler/capital-federal"
INITIAL_URL = f"{BASE_URL}.html"

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

# Viewports realistas
VIEWPORTS = [
    (1920, 1080),
    (1366, 768),
    (1536, 864),
    (1440, 900),
    (1280, 720),
    (1680, 1050),
    (2560, 1440)
]

# URLs de referencia para simular navegación orgánica
REFERERS = [
    "https://www.google.com.ar/search?q=departamentos+alquiler+capital+federal",
    "https://www.google.com.ar/search?q=alquiler+departamento+buenos+aires",
    "https://www.google.com.ar/search?q=argenprop+alquiler",
    "https://www.google.com.ar/",
    "https://www.bing.com/search?q=alquiler+departamento+caba",
    ""  # Direct navigation
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def get_random_viewport():
    return random.choice(VIEWPORTS)

def get_random_referer():
    return random.choice(REFERERS)

def random_sleep(min_seconds=3, max_seconds=8):
    """Sleep for a random amount of time to simulate human behavior"""
    sleep_time = random.uniform(min_seconds, max_seconds)
    sleep(sleep_time)

def clean_browser_data(driver):
    """Limpiar completamente datos del navegador"""
    try:
        # Limpiar localStorage
        driver.execute_script("window.localStorage.clear();")
        
        # Limpiar sessionStorage
        driver.execute_script("window.sessionStorage.clear();")
        
        # Limpiar cookies
        driver.delete_all_cookies()
        
        # Limpiar cache del navegador si es posible
        driver.execute_script("window.indexedDB.deleteDatabase('');")
        
        log("INFO", "Browser data cleaned successfully")
    except Exception as e:
        log("WARNING", f"Error cleaning browser data: {e}")

def advanced_anti_detection(driver):
    """Aplicar técnicas avanzadas anti-detección"""
    try:
        # Eliminar propiedades que revelan automatización
        scripts = [
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})",
            "Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]})",
            "Object.defineProperty(navigator, 'languages', {get: () => ['es-AR', 'es', 'en']})",
            "delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array",
            "delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise",
            "delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol",
            "delete window.chrome.runtime.onConnect",
            "delete window.chrome.runtime.onMessage",
            # Simular propiedades de navegador real
            "Object.defineProperty(screen, 'availTop', {get: () => 0})",
            "Object.defineProperty(screen, 'availLeft', {get: () => 0})",
            # Modificar canvas fingerprinting
            "HTMLCanvasElement.prototype.toDataURL = function() { return 'data:image/png;base64,iVBORw0KGgoAAAANS'; }",
            # Modificar WebGL fingerprinting
            "WebGLRenderingContext.prototype.getParameter = function(parameter) { if (parameter === 37445) return 'Intel Open Source Technology Center'; if (parameter === 37446) return 'Mesa DRI Intel(R) HD Graphics'; return 'Generic'; }",
        ]
        
        for script in scripts:
            try:
                driver.execute_script(script)
            except:
                pass
                
        log("INFO", "Advanced anti-detection applied")
    except Exception as e:
        log("WARNING", f"Error applying anti-detection: {e}")

def simulate_human_behavior(driver):
    """Simulate human-like behavior: scrolling, mouse movements, reading patterns"""
    try:
        # Simular patrón de lectura humano
        reading_actions = [
            # Scroll lento hacia abajo simulando lectura
            lambda: driver.execute_script(f"window.scrollTo(0, {random.randint(100, 400)});"),
            # Pausa para "leer"
            lambda: sleep(random.uniform(2, 5)),
            # Scroll hacia arriba como si revisáramos algo
            lambda: driver.execute_script("window.scrollBy(0, -100);"),
            # Scroll al final para ver más contenido
            lambda: driver.execute_script("window.scrollTo(0, document.body.scrollHeight);"),
            # Volver arriba
            lambda: driver.execute_script("window.scrollTo(0, 0);"),
        ]
        
        # Ejecutar 2-4 acciones aleatorias
        num_actions = random.randint(2, 4)
        selected_actions = random.sample(reading_actions, num_actions)
        
        for action in selected_actions:
            action()
            sleep(random.uniform(0.5, 2.0))
            
        # Random mouse movement más realista
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, "div, p, h1, h2, h3, a")[:10]
            if elements:
                # Mover mouse a 2-3 elementos diferentes
                for _ in range(random.randint(2, 3)):
                    target = random.choice(elements)
                    ActionChains(driver).move_to_element(target).perform()
                    sleep(random.uniform(0.3, 1.0))
        except:
            pass
            
        log("INFO", "Human behavior simulation completed")
    except Exception as e:
        log("WARNING", f"Error in simulating human behavior: {e}")

def set_random_viewport(driver):
    """Cambiar viewport a un tamaño aleatorio realista"""
    try:
        width, height = get_random_viewport()
        driver.set_window_size(width, height)
        log("INFO", f"Viewport set to {width}x{height}")
    except Exception as e:
        log("WARNING", f"Error setting viewport: {e}")

def simulate_organic_navigation(driver, url):
    """Simular navegación orgánica con referer"""
    try:
        referer = get_random_referer()
        if referer:
            # Primero ir al sitio referer
            try:
                driver.get(referer)
                sleep(random.uniform(1, 3))
            except:
                pass
        
        # Ahora navegar a la URL objetivo
        driver.get(url)
        log("INFO", f"Navigated from referer: {referer or 'Direct'}")
    except Exception as e:
        log("WARNING", f"Error in organic navigation: {e}")
        # Fallback a navegación directa
        driver.get(url)

def gen_driver():
    try:
        # Usar un User Agent aleatorio
        user_agent = get_random_user_agent()
        
        log("INFO", f"Creating driver with User Agent: {user_agent[:50]}...")
        
        driver = Driver(
            uc=True, 
            browser="chrome", 
            agent=user_agent, 
            headless=False, 
            undetectable=True, 
            incognito=True
        )
        
        # Aplicar viewport aleatorio
        set_random_viewport(driver)
        
        # Limpiar datos del navegador
        clean_browser_data(driver)
        
        # Aplicar técnicas avanzadas anti-detección
        advanced_anti_detection(driver)
        
        log("INFO", "Driver created successfully with advanced anti-detection")
        return driver
    except Exception as e:
        log("ERROR", f"Error in generating driver: {e}")
        return None

def bs4_parse_raw_html(raw_html):
    return_list = list()
    # Initialize BeautifulSoup
    soup = BeautifulSoup(raw_html, 'html.parser')

    # We look for the container of the property cards
    postings_container = soup.find('div', class_='listing-container')

    # Process each property card
    if postings_container:
        property_cards = soup.select("div[class^=listing__item]")
        # Process each property card
        for idx, card in enumerate(property_cards):
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
    return return_list

def start_scrapping(out_file, iterations, s3_client, bucket_name):
    driver = gen_driver()
    if driver is None:
        log("ERROR", "Failed to create driver. Aborting scraping process.")
        return
    
    out_values = list()
    failed_pages = []
    consecutive_failures = 0
    max_consecutive_failures = 3

    log("INFO", "Starting the scraping process for ArgenProp")

    # Iterar secuencialmente desde la primera página (comportamiento humano natural)
    for page_number in range(1, iterations + 1):
        if page_number == 1:
            url = INITIAL_URL
        else:
            url = f"{BASE_URL}?pagina-{page_number}"
        
        log("INFO", f"Scraping page {page_number} of {iterations} - URL: {url}")

        try:
            # Usar navegación orgánica con referer
            simulate_organic_navigation(driver, url)
            
            # Esperar que la página cargue completamente
            random_sleep(3, 6)
            
            # Simular comportamiento humano más realista
            simulate_human_behavior(driver)
            
        except Exception as e:
            log("WARNING", f"Error in loading URL: {e} - Skipping page {page_number}")
            failed_pages.append(page_number)
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                log("ERROR", f"Too many consecutive failures ({consecutive_failures}). Taking longer break...")
                random_sleep(60, 120)  # Aumentar tiempo de descanso
                consecutive_failures = 0
            continue

        # Verificar si hay CAPTCHA y manejarlo automáticamente
        captcha_attempts = 0
        max_captcha_attempts = 5
        
        while ("confirm you are human" in driver.page_source.lower()) and captcha_attempts < max_captcha_attempts:
            captcha_attempts += 1
            consecutive_failures += 1
            
            # Calcular tiempo de espera progresivo más agresivo (60s, 180s, 360s, 720s, 1200s)
            wait_time = min(60 * (3 ** (captcha_attempts - 1)), 1200)  # Hasta 20 minutos max
            log("WARNING", f"CAPTCHA detected (attempt {captcha_attempts}/{max_captcha_attempts}). Waiting {wait_time//60:.1f} minutes and regenerating driver with new identity...")
            
            # Cerrar driver actual completamente
            try:
                driver.quit()
            except:
                pass
            
            # Esperar tiempo progresivo más largo
            random_sleep(wait_time, wait_time + 60)
            
            # Regenerar driver con identidad completamente nueva
            driver = gen_driver()
            if driver is None:
                log("ERROR", "Failed to regenerate driver. Aborting scraping process.")
                return
            
            # Simular navegación más humana después de CAPTCHA
            try:
                # Primero visitar página principal o de búsqueda
                if random.random() < 0.5:
                    driver.get("https://www.argenprop.com.ar")
                    random_sleep(3, 6)
                    simulate_human_behavior(driver)
                
                # Luego navegar a la URL objetivo
                simulate_organic_navigation(driver, url)
                random_sleep(4, 8)
                simulate_human_behavior(driver)
            except Exception as e:
                log("WARNING", f"Error reloading page after CAPTCHA: {e}")
                continue
        
        # Si después de todos los intentos sigue habiendo CAPTCHA, saltar esta página
        if captcha_attempts >= max_captcha_attempts and ("confirm you are human" in driver.page_source.lower() or "captcha" in driver.page_source.lower()):
            log("ERROR", f"CAPTCHA persists after {max_captcha_attempts} attempts. Skipping page {page_number}")
            failed_pages.append(page_number)
            consecutive_failures += 1
            
            # Si hay muchos fallos consecutivos, tomar un descanso muy largo
            if consecutive_failures >= max_consecutive_failures:
                log("WARNING", f"Too many consecutive CAPTCHA failures. Taking extended break (10-15 minutes)...")
                random_sleep(600, 900)  # 10-15 minutos
                consecutive_failures = 0
            continue

        # Resetear contador de fallos consecutivos si llegamos aquí
        consecutive_failures = 0

        # Espera aleatoria más larga entre páginas (aumentada para evitar detección)
        random_sleep(12, 25)

        # We delegate html analysis to BeautifulSoup
        raw_html = driver.page_source

        try:
            extracted_values = bs4_parse_raw_html(raw_html)
            log("INFO", f"Extracted {len(extracted_values)} property cards from page {page_number}")
            if not extracted_values:
                log("WARNING", f"No property cards found on page {page_number}")
                failed_pages.append(page_number)
            else:
                out_values.extend(extracted_values)
        except Exception as e:
            log("WARNING", f"Error in extracting values: {e} - Skipping page {page_number}")
            failed_pages.append(page_number)
            continue

        # Cada 20 páginas, tomar un descanso más largo
        if page_number % 20 == 0:
            log("INFO", f"Taking extended break after {page_number} pages...")
            random_sleep(60, 120)

    # Retry failed pages with longer delays
    if failed_pages:
        log("INFO", f"Retrying {len(failed_pages)} failed pages with longer delays...")
        for page_number in failed_pages[:min(10, len(failed_pages))]:  # Retry max 10 failed pages
            if page_number == 1:
                url = INITIAL_URL
            else:
                url = f"{BASE_URL}?pagina-{page_number}"
            
            log("INFO", f"Retrying page {page_number} with enhanced stealth...")
            random_sleep(20, 35)  # Longer delay for retries
            
            try:
                # Aplicar limpieza completa antes del retry
                clean_browser_data(driver)
                advanced_anti_detection(driver)
                
                # Usar navegación orgánica
                simulate_organic_navigation(driver, url)
                random_sleep(5, 10)
                simulate_human_behavior(driver)
                
                raw_html = driver.page_source
                
                # Verificar si hay CAPTCHA en el retry
                if "confirm you are human" in raw_html.lower() or "captcha" in raw_html.lower():
                    log("WARNING", f"CAPTCHA detected on retry for page {page_number}. Skipping.")
                    continue
                
                extracted_values = bs4_parse_raw_html(raw_html)
                if extracted_values:
                    out_values.extend(extracted_values)
                    log("INFO", f"Successfully retried page {page_number}")
            except Exception as e:
                log("WARNING", f"Retry failed for page {page_number}: {e}")

    log("INFO", "Scraping process completed! Generating STOCK file...")
    
    if out_values:
        df = pd.DataFrame(out_values)
        # Eliminar duplicados basados en argenprop_code
        df = df.drop_duplicates(subset=['argenprop_code'], keep='first')
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        s3_client.put_object(Bucket=bucket_name, Key=out_file, Body=csv_buffer.getvalue())
        log("INFO", f"Successfully saved {len(df)} unique properties to S3")
    else:
        log("ERROR", "No data was collected!")

    # Cerrar driver de manera segura
    try:
        if driver:
            driver.quit()
    except Exception as e:
        log("WARNING", f"Error closing driver: {e}") 