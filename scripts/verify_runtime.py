"""Check HTTP servers and a real Streamlit form submission."""
import json
import socket
import subprocess
import sys
import time

import requests
from streamlit.testing.v1 import AppTest

from src.inference import predict_record
from src.util import ROOT, path


def wait_for(url, process, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Service exited before readiness: {url}')
        try:
            response = requests.get(url, timeout=2)
            if response.ok:
                return response
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError(f'Service did not become healthy: {url}')


def main():
    example = json.loads(path('reports/example_request.json').read_text(encoding='utf-8'))
    for port in (8080, 8501):
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'Port {port} is already occupied; runtime check requires its own services')
    api = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'src.api:app', '--host', '127.0.0.1', '--port', '8080'], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    ui = None
    try:
        wait_for('http://127.0.0.1:8080/health', api)
        assert wait_for('http://127.0.0.1:8080/ready', api).json()['status'] == 'ready'
        response = requests.post('http://127.0.0.1:8080/predict', json=example, timeout=15)
        response.raise_for_status()
        direct = predict_record(example)
        assert response.json() == direct
        assert requests.post('http://127.0.0.1:8080/predict', json={}, timeout=15).status_code == 422
        assert requests.post('http://127.0.0.1:8080/predict', json={**example, 'injury_claim': 'invalid'}, timeout=15).status_code == 422
        supported_missing = {**example, 'injury_claim': None, 'collision_type': '?', 'auto_make': 'UNSEEN MAKE'}
        missing_response = requests.post('http://127.0.0.1:8080/predict', json=supported_missing, timeout=15)
        missing_response.raise_for_status()
        assert missing_response.json() == predict_record(supported_missing)
        print('PASS API startup, health, readiness, valid/invalid/missing/unseen requests, direct score match', flush=True)
        ui = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'src/dashboard.py', '--server.headless=true', '--server.address=127.0.0.1', '--server.port=8501'], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        wait_for('http://127.0.0.1:8501/_stcore/health', ui)
        test = AppTest.from_file(str(path('src/dashboard.py')), default_timeout=30).run()
        assert not test.exception, test.exception
        test.button[0].click().run()
        assert not test.exception, test.exception
        assert test.metric and float(test.metric[0].value) == round(direct['fraud_score'], 3)
        print('PASS Streamlit startup and real form submission through API', flush=True)
    finally:
        for process in (ui, api):
            if process is not None:
                process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.communicate()


if __name__ == '__main__':
    main()
