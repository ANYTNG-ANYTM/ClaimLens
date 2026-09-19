"""Streamlit's prediction transport, kept testable without browser automation."""
import requests


def request_prediction(claim, api_url='http://localhost:8080', transport=requests):
    response = transport.post(f'{api_url.rstrip("/")}/predict', json=claim, timeout=15)
    response.raise_for_status()
    return response.json()
