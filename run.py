
import uvicorn

if __name__ == "__main__":
    print("Starte OCR-Server...")
    print("Frontend: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )