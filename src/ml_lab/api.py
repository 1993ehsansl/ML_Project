"""Inference API; run training first to generate model artifacts."""
import os
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title='ML Foundations to Production', version='0.1.0')
MODEL_DIR = Path(os.getenv('MODEL_DIR', 'artifacts'))


class DigitInput(BaseModel):
    pixels: list[float] = Field(min_length=64, max_length=64)


class RegressionInput(BaseModel):
    x: float


@lru_cache(maxsize=8)
def load_model(name):
    path = MODEL_DIR/f'{name}.joblib'
    if not path.exists():
        raise FileNotFoundError(f'Train models first; missing {path}')
    return joblib.load(path)


@app.get('/health')
def health():
    return {'status': 'ok', 'models_available': sorted(p.stem for p in MODEL_DIR.glob('*.joblib'))}


@app.post('/predict/digit')
def predict_digit(item: DigitInput, model: str = 'svm'):
    if model not in ('svm', 'logistic_regression'):
        raise HTTPException(400, 'model must be svm or logistic_regression')
    if not np.isfinite(item.pixels).all() or any(p < 0 or p > 16 for p in item.pixels):
        raise HTTPException(422, 'pixels must be finite and within [0,16]')
    try:
        classifier = load_model(model)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    arr = np.asarray(item.pixels, dtype=float)[None, :]
    proba = classifier.predict_proba(arr)[0]
    return {'digit': int(classifier.classes_[proba.argmax()]),
            'confidence': float(proba.max()), 'probabilities': proba.tolist(), 'model': model}


@app.post('/predict/regression')
def predict_regression(item: RegressionInput):
    if not np.isfinite(item.x):
        raise HTTPException(422, 'x must be finite')
    try:
        model = load_model('bayesian')
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    mean, std = model.predict([[item.x]], return_std=True)
    return {'mean': float(mean[0]), 'std': float(std[0]),
            'lower_95': float(mean[0]-1.96*std[0]),
            'upper_95': float(mean[0]+1.96*std[0])}
