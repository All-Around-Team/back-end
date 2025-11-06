import io, cv2, numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

from gemini import get_velocity_async
from models import ScanRequest, ScanResponse
from utils.gpt import gpt_ocr
from utils.classifier import classify_text

app = FastAPI()


@app.post("/validate", response_model=ScanResponse)
async def validate(req: ScanRequest):
    label, score = await classify_text(req.text)
    safe = label == "appropriate"
    return ScanResponse(
        safe=safe,
        label=label.upper(),
        score=float(score),
        ocr_text=""
    )


@app.post("/validate-multiple", response_model=list[ScanResponse])
async def validate_multiple(reqs: list[ScanRequest]):
    out = []
    for r in reqs:
        label, score = await classify_text(r.text)
        safe = label == "appropriate"
        out.append(ScanResponse(
            safe=safe,
            label=label.upper(),
            score=float(score),
            ocr_text=""
        ))
    return out


@app.post("/validate-image", response_model=ScanResponse)
async def validate_image(file: UploadFile = File(...)):
    img = await file.read()
    img = preprocess(Image.open(io.BytesIO(img)))

    text = await gpt_ocr(img)

    if not text.strip():
        return ScanResponse(safe=True, label="NO_TEXT", score=0.0, ocr_text="")

    label, score = await classify_text(text)
    safe = label == "appropriate"

    return ScanResponse(
        safe=safe,
        label=label.upper(),
        score=float(score),
        ocr_text=text
    )


@app.post("/validate-gemini", response_model=list[float])
async def validate_gemini(requests: list[ScanRequest]):
    if not requests:
        raise HTTPException(status_code=400, detail="Empty request list")

    texts = [r.text for r in requests]

    velocities, err = await get_velocity_async(texts)
    if err is not None:
        raise HTTPException(status_code=502, detail=err)

    return velocities


def preprocess(img: Image.Image) -> bytes:
    img = img.convert("RGB")
    arr = np.array(img)

    h, w = arr.shape[:2]
    scale = 1600 / max(h, w) if max(h, w) > 1600 else 1
    arr = cv2.resize(arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    sharp = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.8, sharp, -0.8, 0)

    _, buf = cv2.imencode(".jpg", sharp)
    return buf.tobytes()
