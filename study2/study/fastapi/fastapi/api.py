import requests as requests
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def get_api_call():
    # Make an API call using the requests library
    response = requests.get("http://172.16.100.102:8000/openapi.json")

    # Get the JSON data from the response
    data = response.json()

    # Return the data to the client
    return data