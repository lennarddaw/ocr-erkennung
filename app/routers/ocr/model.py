from typing import Optional

from pydantic import BaseModel


class ImageBase64(BaseModel):
    img_body_base64: str
    filename: Optional[str] = None
