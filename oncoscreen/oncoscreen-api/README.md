# OncoScreen API

FastAPI service that serves the model trained in [oncoscreen-ml](../oncoscreen-ml).

> Educational project. Not a medical device.

## Setup

1. Train the model in `oncoscreen-ml` and copy `model.joblib` into this repo's root.
2. Run:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Docs: http://127.0.0.1:8000/docs

## Endpoints

| Method | Path | What it does |
|--------|------|--------------|
| GET | `/health` | Liveness check |
| GET | `/features` | The 30 feature names, in order |
| POST | `/predict` | Body `{"features": [30 floats]}` returns label + confidence |

## Docker

```bash
docker build -t oncoscreen-api .
docker run -p 8000:8000 oncoscreen-api
```

## Test

```bash
pytest
```

(Prediction tests are skipped if `model.joblib` is missing.)

## License

MIT
