from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

bundle = joblib.load(Path(__file__).resolve().parent.parent / "model.joblib")
app = FastAPI(title="OncoScreen API", version="1.0.0")


class Sample(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def features():
    return bundle["features"]


@app.post("/predict")
def predict(sample: Sample):
    n = len(bundle["features"])
    if len(sample.features) != n:
        raise HTTPException(status_code=422, detail=f"expected {n} features")
    probs = bundle["model"].predict_proba([sample.features])[0]
    i = int(probs.argmax())
    return {"prediction": bundle["classes"][i], "confidence": round(float(probs[i]), 4)}
