from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
import undetected_chromedriver as uc
from selenium_stealth import stealth
from seleniumbase import Driver

### Custom Imports
from utils import log

DOMAIN_URL = "https://www.argenprop.com.ar"
BASE_URL = "https://www.argenprop.com/departamentos/alquiler/capital-federal"
INITIAL_URL = f"{BASE_URL}.html"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"


def gen_driver():
    try:
        driver = Driver(uc=True, browser="chrome", agent=USER_AGENT, headless=False, undetectable=True, incognito=True)
        return driver
    except Exception as e:
        log("ERROR", f"Error in generating driver: {e}")

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

def start_scrapping(out_file, iterations):
    driver = gen_driver()
    out_values = list()

    log("INFO", "Starting the scraping process for ArgenProp")

    # Scrape the initial page and 500 additional pages
    for page_number in range(1, iterations+1):  # Pages 1 to 500
        if page_number == 1:
            url = INITIAL_URL
        else:
            url = f"{BASE_URL}?pagina-{page_number}"
        log("INFO", f"Scraping page {page_number} of {iterations} - URL: {url}")

        try:
            driver.get(url)
        except Exception as e:
            log("WARNING", f"Error in loading URL: {e} - Skipping page {page_number}")
            continue

        if "confirm you are human" in driver.page_source.lower():
            log("WARNING", "CAPTCHA detected. Please solve it manually...")
            input("Press Enter after solving the CAPTCHA...")

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
    df.to_csv(out_file, index=False)
    driver.quit()