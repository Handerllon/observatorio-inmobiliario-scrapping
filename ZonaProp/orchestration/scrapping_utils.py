### System Imports
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
import undetected_chromedriver as uc
from seleniumbase import Driver

### Custom Imports
from utils import log
from io import StringIO

DOMAIN_URL = "https://www.zonaprop.com.ar"
BASE_URL = "https://www.zonaprop.com.ar/departamentos-alquiler-capital-federal"
INITIAL_URL = f"{BASE_URL}.html"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"

def gen_driver():
    try:
        driver = Driver(uc=True, browser="chrome", agent=USER_AGENT, headless=True, undetectable=True, incognito=True)
        return driver
    except Exception as e:
        log("ERROR", f"Error in generating driver: {e}")

def bs4_parse_raw_html(raw_html):
    return_list = list()
    # Initialize BeautifulSoup
    soup = BeautifulSoup(raw_html, 'html.parser')

    # We look for the container of the property cards
    postings_container = soup.find('div', class_='postings-container')

    # Process each property card
    if postings_container:
        property_cards = soup.select("div[class^=CardContainer]")
        for idx, card in enumerate(property_cards):
            property_url = DOMAIN_URL + card.find('div', {'data-qa': 'posting PROPERTY'}).get('data-to-posting')
            price = card.find('div', {'data-qa': 'POSTING_CARD_PRICE'}).text.strip() if card.find('div', {'data-qa': 'POSTING_CARD_PRICE'}) else None
            expenses = card.find('div', {'data-qa': 'expensas'}).text.strip() if card.find('div', {'data-qa': 'expensas'}) else None
            address = card.select_one('div[class*="LocationAddress"]').text.strip() if card.select_one('div[class*="LocationAddress"]') else None
            location = card.find('h2', {'data-qa': 'POSTING_CARD_LOCATION'}).text.strip() if card.find('h2', {'data-qa': 'POSTING_CARD_LOCATION'}) else None
            features = card.find('h3', {'data-qa': 'POSTING_CARD_FEATURES'}).text.strip() if card.find('h3', {'data-qa': 'POSTING_CARD_FEATURES'}) else None
            description = card.find('h3', {'data-qa': 'POSTING_CARD_DESCRIPTION'}).text.strip() if card.find('h3', {'data-qa': 'POSTING_CARD_DESCRIPTION'}) else None
            return_list.append({
                'property_url': property_url,
                'zonaprop_code': property_url.split("-")[-1].split(".")[0],
                'price': price,
                'expenses': expenses,
                'address': address,
                'location': location,
                'features': features,
                'description': description
            })
    return return_list

def start_scrapping(out_file, iterations, s3_client, bucket_name):
    driver = gen_driver()
    out_values = list()

    log("INFO", "Starting the scraping process for ZonaProp")

    # Scrape the initial page and 500 additional pages
    for page_number in range(1, iterations+1):  # Pages 1 to 500
        if page_number == 1:
            url = INITIAL_URL
        else:
            url = f"{BASE_URL}-pagina-{page_number}.html"
        log("INFO", f"Scraping page {page_number} of {iterations} - URL: {url}")

        try:
            driver.get(url)
        except Exception as e:
            log("WARNING", f"Error in loading URL: {e} - Skipping page {page_number}")
            continue

        # Sleep for 5 seconds to reduce the chance of being detected
        sleep(5)

        # We delegate html analysis to BeautifulSoup
        raw_html = driver.page_source

        try:
            extracted_values = bs4_parse_raw_html(raw_html)
            log("INFO", f"Extracted {len(extracted_values)} property cards from page {page_number}")
            if not extracted_values:
                log("WARNING", f"No property cards found on page {page_number}")
            out_values.extend(extracted_values)
        except Exception as e:
            log("WARNING", f"Error in extracting values: {e} - Skipping page {page_number}")
            continue

        out_values.extend(extracted_values)

    log("INFO", "Scraping process completed! Generating STOCK file...")
    df = pd.DataFrame(out_values)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    s3_client.put_object(Bucket=bucket_name, Key=out_file, Body=csv_buffer.getvalue())

    driver.quit()