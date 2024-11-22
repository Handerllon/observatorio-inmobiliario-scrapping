from bs4 import BeautifulSoup

with open("output/test_output.html") as f:
    raw_html = f.read()
    soup = BeautifulSoup(raw_html, 'html.parser')

return_list = list()

postings_container = soup.find('div', class_='listing-container')
if postings_container:
    property_cards = soup.select("div[class^=listing__item]")
    print(f"Found {len(property_cards)} property cards")
    # Process each property card
    for idx, card in enumerate(property_cards):
        property_url = "https://www.argenprop.com" + card.find('a', class_='card')['href']
        currency = card.find('p', class_='card__price').find('span', class_='card__currency').text.strip() if card.find('p', class_='card__price').find('span', class_='card__currency').text.strip() else None
        only_price = card.find('p', class_='card__price').find('span', class_='card__currency').next_sibling.strip() if card.find('p', class_='card__price').find('span', class_='card__currency').next_sibling.strip() else None
        price = currency + " " + only_price
        expenses = card.find('span', class_='card__expenses').text.strip().replace("+ ", "") if card.find('span', class_='card__expenses') else None
        address = card.find('p', class_='card__address').text.strip() if card.find('p', class_='card__address').text.strip() else None
        location = card.find('p', class_='card__title--primary').text.strip() if card.find('p', class_='card__title--primary').text.strip() else None
        features = soup.find('ul', class_='card__main-features')
        features_text = [li.text.strip() for li in features.find_all('li')] if features else None
        description = card.find('h2', class_='card__title').text.strip() if card.find('h2', class_='card__title').text.strip() else None
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