import json

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.data_pipeline import dataset_sha256, prepare
from src.inference import load_bundle, predict_record
from src.ui_client import request_prediction
from src.util import load_config, path


@pytest.fixture(scope='module')
def sample():
    return json.loads(path('reports/example_request.json').read_text(encoding='utf-8'))


def test_split_and_raw_checksum():
    parts, audit = prepare(save=False)
    ids = [set(p.policy_number) for p in parts.values()]
    assert len(set.union(*ids)) == sum(map(len, ids)) == 1000
    assert audit['sha256'] == dataset_sha256()


def test_saved_pipeline_excludes_target_and_consistent_api_ui(sample):
    bundle = load_bundle()
    assert load_bundle() is bundle
    assert 'fraud_reported' not in bundle['features']
    direct = predict_record(sample)
    client = TestClient(app)
    assert client.get('/health').status_code == 200
    assert client.get('/ready').status_code == 200
    response = client.post('/predict', json=sample)
    assert response.status_code == 200
    via_ui = request_prediction(sample, 'http://testserver', transport=client)
    assert direct == response.json() == via_ui


def test_bundle_metadata_matches_release_results():
    bundle = load_bundle()
    results = json.loads(path('reports/metrics/model_results.json').read_text(encoding='utf-8'))
    config = load_config()
    assert bundle['threshold'] == results['threshold']
    assert bundle['model_name'] == results['selected_model']
    assert bundle['dataset_sha256'] == results['dataset_sha256']
    assert bundle['features'] == config['numeric_features'] + config['categorical_features']
    assert list(bundle['pipeline'].classes_) == [0, 1]


def test_missing_unseen_invalid_and_threshold(sample):
    client = TestClient(app)
    unseen = dict(sample, auto_make='A NEW BRAND', collision_type='?', property_damage=None, injury_claim=None)
    assert client.post('/predict', json=unseen).status_code == 200
    assert client.post('/predict', json={}).status_code == 422
    assert client.post('/predict', json=dict(sample, injury_claim='oops')).status_code == 422
    assert client.post('/predict', json=dict(sample, injury_claim='inf')).status_code == 422
    bundle = dict(load_bundle())
    bundle['threshold'] = 0.0
    assert predict_record(sample, bundle)['prediction'] == 'Y'
    bundle['threshold'] = 1.0
    assert predict_record(sample, bundle)['prediction'] == 'N'
