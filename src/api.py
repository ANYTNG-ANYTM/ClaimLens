"""HTTP interface for one claim at a time."""
from fastapi import Body, FastAPI, HTTPException

from .inference import load_bundle, predict_record

app = FastAPI(title='Insurance claim review model')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/ready')
def ready():
    try:
        bundle = load_bundle()
        return {'status': 'ready', 'model': bundle['model_name']}
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post('/predict')
def predict(claim: dict = Body(...)):
    try:
        return predict_record(claim)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
