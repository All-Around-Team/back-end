from fastapi import FastAPI

from models import ScanResponse, ScanRequest
from example import classifier

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/validate", response_model=ScanResponse)
async def validate(request: ScanRequest):
    result = classifier(request.text)[0]

    label = result["label"]
    score = float(result["score"])

    safe = not (label == "INJECTION" and score > 0.5)

    return ScanResponse(
        safe=safe,
        label=label,
        score=score,
    )

@app.post("/validate-multiple", response_model=list[ScanResponse])
async def validate_multiple(request: list[ScanRequest]):
    responses = []
    for req in request:
        result = classifier(req.text)[0]

        label = result["label"]
        score = float(result["score"])

        safe = not (label == "INJECTION" and score > 0.5)

        responses.append(
            ScanResponse(
                safe=safe,
                label=label,
                score=score,
            )
        )
    return responses