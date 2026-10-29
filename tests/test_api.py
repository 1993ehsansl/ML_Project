from fastapi.testclient import TestClient
from ml_lab.api import app


def test_health():
    response = TestClient(app).get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_invalid_digit():
    response = TestClient(app).post('/predict/digit', json={'pixels': [0]*63})
    assert response.status_code == 422
