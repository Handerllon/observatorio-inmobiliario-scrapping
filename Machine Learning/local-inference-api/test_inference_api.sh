curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "antiquity": 20,
  "bedrooms": 1,
  "rooms": 2,
  "total_area": 60
}'
