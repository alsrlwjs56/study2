from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
app = FastAPI()


db = []

class City(BaseModel) :
	name : str
	timezone : str


@app.get("/")
async def root():
	return { "message" : "Hello Funzin" }
