"""FastAPI app: handwritten digit recognition API."""
import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "ml/registry/model.pkl"

app = FastAPI(title="Digit Recognition API")


class PredictRequest(BaseModel):
    pixels: list[float] = Field(..., min_length=64, max_length=64)


def _load_model():
    return joblib.load(MODEL_PATH)


model = _load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if any(p < 0 or p > 16 for p in req.pixels):
        raise HTTPException(status_code=422, detail="pixels must be in [0, 16]")
    X = np.array(req.pixels).reshape(1, -1)
    pred = int(model.predict(X)[0])
    return {"prediction": pred}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
