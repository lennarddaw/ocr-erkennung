from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routers import ocr

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ocr.router)

@app.get("/")
async def root():
    """
    Serviert die index.html Seite
    """
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")