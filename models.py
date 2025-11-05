from pydantic import BaseModel


class ScanRequest(BaseModel):
    text: str


class ScanResponse(BaseModel):
    safe: bool
    label: str
    score: float
    ocr_text: str | None = None
