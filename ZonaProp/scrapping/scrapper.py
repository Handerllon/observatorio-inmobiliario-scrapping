from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
import undetected_chromedriver as uc
from selenium_stealth import stealth
from seleniumbase import Driver

OUT_FILE = "stock_zonaprop_18012025.csv"

def gen_driver():
    try:
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        #driver = uc.Chrome(options=chrome_options)
        driver = Driver(uc=True, browser="chrome", agent=user_agent, headless=True)
        return driver
    except Exception as e:
        print("Error in Driver: ",e)

def bs4_parse_raw_html(raw_html):
    print("Parsing raw HTML...")
    return_list = list()
    # Initialize BeautifulSoup
    soup = BeautifulSoup(raw_html, 'html.parser')

    # We look for the container of the property cards
    postings_container = soup.find('div', class_='postings-container')

    # Process each property card
    if postings_container:
        property_cards = soup.select("div[class^=CardContainer]")
        print(f"Found {len(property_cards)} property cards on this URL")
        for idx, card in enumerate(property_cards):
            property_url = "https://www.zonaprop.com.ar" + card.find('div', {'data-qa': 'posting PROPERTY'}).get('data-to-posting')
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

    
    for value in return_list:
        print(value["property_url"])
    return return_list


driver = gen_driver()
out_values = list()

print("Starting the scraping process...")
try:
    base_url = "https://www.zonaprop.com.ar/departamentos-alquiler-capital-federal"
    initial_url = f"{base_url}.html"

    # Scrape the initial page and 500 additional pages
    for page_number in range(1, 501):  # Pages 1 to 500
        if page_number == 1:
            url = initial_url
        else:
            url = f"{base_url}-pagina-{page_number}.html"

        print(f"Scraping: {url}")
        driver.get(url)

        # Check for CAPTCHA, we handle it manually
        if "verifying you are human. this may take a few seconds." in driver.page_source.lower():
            print("CAPTCHA detected. Please solve it manually.")
            input("Press Enter after solving the CAPTCHA...")

        # Sleep for 5 seconds to reduce the chance of being detected
        print("Sleeping for 5 seconds")
        sleep(5)

        # We delegate html analysis to BeautifulSoup
        print("Extracting property cards...")
        raw_html = driver.page_source
        out_values.extend(bs4_parse_raw_html(raw_html))

finally:
    df = pd.DataFrame(out_values)
    df.to_csv("output/{}".format(OUT_FILE), index=False)
    driver.quit()