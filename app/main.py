from fastapi import FastAPI
from app.routers import ocr

app = FastAPI()

app.include_router(ocr.router)

@app.get("/")
async def root():
    return {"message": "Hello Hackernoon.com!"}