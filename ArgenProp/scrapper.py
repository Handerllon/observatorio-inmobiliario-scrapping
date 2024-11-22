from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
import undetected_chromedriver as uc
from selenium_stealth import stealth

def gen_driver():
    try:
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.6167.140 Safari/537.36"
        chrome_options = uc.ChromeOptions()
        #chrome_options.add_argument('--headless=new')
        #chrome_options.add_argument("--start-maximized")
        #chrome_options.add_argument("--blink-settings=imagesEnabled=false")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("user-agent={}".format(user_agent))
        driver = uc.Chrome(options=chrome_options)
        stealth(driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True
        )
        return driver
    except Exception as e:
        print("Error in Driver: ",e)

def bs4_parse_raw_html(raw_html):
    print("Parsing raw HTML...")
    return_list = list()
    # Initialize BeautifulSoup
    soup = BeautifulSoup(raw_html, 'html.parser')

    # We look for the container of the property cards
    postings_container = soup.find('div', class_='listing-container')

    # Process each property card
    if postings_container:
        property_cards = soup.select("div[class^=listing__item]")
        print(f"Found {len(property_cards)} property cards")
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
                'zonaprop_code': property_url.split("-")[-1],
                'price': price,
                'expenses': expenses,
                'address': address,
                'location': location,
                'features': features_text,
                'description': description
            })
    return return_list


driver = gen_driver()
out_values = list()

print("Starting the scraping process...")
try:
    base_url = "https://www.argenprop.com/departamentos/alquiler/capital-federal"
    initial_url = f"{base_url}"

    # Scrape the initial page and 500 additional pages
    for page_number in range(1, 501):  # Pages 1 to 500
        if page_number == 1:
            url = initial_url
        else:
            url = f"{base_url}?pagina-{page_number}"

        print(f"Scraping: {url}")
        driver.get(url)

        # Check for CAPTCHA, we handle it manually
        if "confirm you are human" in driver.page_source.lower():
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
    df.to_csv("output/output.csv", index=False)
    driver.quit()