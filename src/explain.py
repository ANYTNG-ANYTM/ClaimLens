"""Coefficient summary for the current fitted logistic model."""
import json
import numpy as np
from .inference import load_bundle
from .util import path


def run():
    pipeline = load_bundle()['pipeline']
    model = pipeline.named_steps['model']
    if not hasattr(model, 'coef_'):
        raise ValueError('Current model has no linear coefficients')
    names = pipeline.named_steps['preprocess'].get_feature_names_out()
    coefficients = model.coef_[0]
    order = np.argsort(np.abs(coefficients))[::-1][:20]
    result = [{'feature': str(names[i]), 'coefficient': float(coefficients[i])} for i in order]
    path('reports/metrics/coefficients.json').write_text(json.dumps({'note': 'Conditional associations of transformed features; not causal effects.', 'top_absolute_coefficients': result}, indent=2), encoding='utf-8')
    print(json.dumps(result[:10], indent=2))
    return result


if __name__ == '__main__':
    run()
