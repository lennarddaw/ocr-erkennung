from fastapi import APIRouter, HTTPException
from app.routers.ocr.model import ImageBase64
import base64

router = APIRouter(
    prefix="/ocr",
    tags=["ocr"]
)

@router.post("/upload-image")
async def upload_image(image_data: ImageBase64):
    try:
        img_bytes = base64.b64decode(image_data.img_body_base64)
        
        return {
            "success": True,
            "message": "Bild erfolgreich empfangen",
            "image_size": len(img_bytes),
            "image_base64": image_data.img_body_base64
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fehler beim Verarbeiten des Bildes: {str(e)}")