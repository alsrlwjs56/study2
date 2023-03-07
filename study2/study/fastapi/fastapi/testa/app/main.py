import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {"hello root"}

@app.get("/world")
def world():
    return {"hello world"}