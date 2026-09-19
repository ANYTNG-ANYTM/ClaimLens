"""Trusted local artifact loading and common prediction path."""
from functools import lru_cache
import joblib
import pandas as pd
from .preprocessing import features
from .util import path


@lru_cache(maxsize=1)
def load_bundle():
    artifact = path('models/fraud_pipeline.joblib')
    if not artifact.exists():
        raise FileNotFoundError(f'Missing {artifact}. Run python -m src.modelling first.')
    try:
        bundle = joblib.load(artifact)  # Only load locally generated, trusted joblib files.
    except Exception as exc:
        raise RuntimeError(f'Could not load {artifact}; rerun python -m src.modelling with current dependencies.') from exc
    if not {'pipeline','threshold','features','positive_class'}.issubset(bundle):
        raise ValueError('Incompatible inference artifact. Rerun python -m src.modelling.')
    return bundle


def predict_record(record, bundle=None):
    bundle = bundle or load_bundle()
    frame = pd.DataFrame([record])
    transformed = features(frame, bundle)
    score = float(bundle['pipeline'].predict_proba(transformed)[0, list(bundle['pipeline'].classes_).index(bundle['positive_class'])])
    flagged = score >= bundle['threshold']
    return {'prediction': 'Y' if flagged else 'N', 'fraud_score': score, 'threshold': bundle['threshold'],
            'review_recommended': bool(flagged)}
