import requests
import io
KERAS_REST_API_URL = "http://localhost:5000/predict"

r = requests.post(KERAS_REST_API_URL,data="I love you").json()

print(r)


