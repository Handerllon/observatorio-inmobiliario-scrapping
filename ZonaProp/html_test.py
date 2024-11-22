from bs4 import BeautifulSoup

with open("output.html") as f:
    raw_html = f.read()
    soup = BeautifulSoup(raw_html, 'html.parser')

return_list = list()

print("Parsing raw HTML...")
property_cards = soup.select("div[class^=CardContainer]")
print(f"Found {len(property_cards)} property cards")
# Process each property card
for idx, card in enumerate(property_cards):
    print(f"Processing card {idx + 1} of {len(property_cards)}")
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
    print(address)