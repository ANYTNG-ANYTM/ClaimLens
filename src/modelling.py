"""Model selection on training folds, threshold selection on validation, one final test."""
import json
import platform
from time import perf_counter
import importlib.metadata as metadata

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, fbeta_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .data_pipeline import prepare
from .preprocessing import features, make_preprocessor
from .util import load_config, path


def candidates(seed):
    return {
        'dummy_prior': (DummyClassifier(strategy='prior'), False),
        'logistic': (LogisticRegression(max_iter=1500, random_state=seed), True),
        'logistic_balanced': (LogisticRegression(max_iter=1500, class_weight='balanced', random_state=seed), True),
        'random_forest': (RandomForestClassifier(n_estimators=150, min_samples_leaf=3, n_jobs=1, random_state=seed), False),
        'random_forest_balanced': (RandomForestClassifier(n_estimators=150, min_samples_leaf=3, class_weight='balanced_subsample', n_jobs=1, random_state=seed), False),
        'hist_gradient_boosting': (HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, learning_rate=0.05, random_state=seed), False),
    }


def assess(y, score, threshold):
    prediction = score >= threshold
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    return {
        'threshold': float(threshold), 'precision_fraud': float(precision_score(y, prediction, zero_division=0)),
        'recall_fraud': float(recall_score(y, prediction, zero_division=0)),
        'f1_fraud': float(f1_score(y, prediction, zero_division=0)),
        'f2_fraud': float(fbeta_score(y, prediction, beta=2, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, score)),
        'average_precision': float(average_precision_score(y, score)),
        'confusion_matrix_N_Y': matrix.tolist(),
        'support': {'N': int((y == 0).sum()), 'Y': int((y == 1).sum())},
    }


def choose_threshold(y, score):
    # An explicit finite grid avoids choosing a threshold from the held-out test set.
    grid = np.arange(0.05, 0.951, 0.01)
    return float(max(grid, key=lambda t: (fbeta_score(y, score >= t, beta=2, zero_division=0), precision_score(y, score >= t, zero_division=0), t)))


def train():
    config = load_config()
    parts, audit = prepare()
    x_train = features(parts['train'], config)
    y_train = parts['train'][config['target']].eq('Y').astype(int)
    x_valid = features(parts['validation'], config)
    y_valid = parts['validation'][config['target']].eq('Y').astype(int)
    x_test = features(parts['test'], config)
    y_test = parts['test'][config['target']].eq('Y').astype(int)
    cv = StratifiedKFold(n_splits=config['cv_folds'], shuffle=True, random_state=config['seed'])
    comparisons = {}
    for name, (classifier, scale) in candidates(config['seed']).items():
        pipeline = Pipeline([('preprocess', make_preprocessor(config, scale)), ('model', classifier)])
        started = perf_counter()
        scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring=config['selection_metric'], n_jobs=1)
        comparisons[name] = {'cv_average_precision_mean': float(scores.mean()),
                             'cv_average_precision_std': float(scores.std(ddof=1)),
                             'cv_folds': scores.tolist(), 'cv_time_seconds': round(perf_counter() - started, 3)}
        print(name, comparisons[name]['cv_average_precision_mean'], flush=True)
    selected = max(comparisons, key=lambda name: comparisons[name]['cv_average_precision_mean'])
    classifier, scale = candidates(config['seed'])[selected]
    pipeline = Pipeline([('preprocess', make_preprocessor(config, scale)), ('model', classifier)])
    started = perf_counter()
    pipeline.fit(x_train, y_train)
    fit_seconds = perf_counter() - started
    validation_score = pipeline.predict_proba(x_valid)[:, list(pipeline.classes_).index(1)]
    threshold = choose_threshold(y_valid, validation_score)
    # Freeze fitted pipeline and threshold before accessing test labels or scores.
    bundle = {'pipeline': pipeline, 'threshold': threshold, 'features': config['numeric_features'] + config['categorical_features'],
              'numeric_features': config['numeric_features'], 'categorical_features': config['categorical_features'],
              'positive_class': 1, 'model_name': selected, 'dataset_sha256': audit['sha256']}
    path('models').mkdir(exist_ok=True)
    joblib.dump(bundle, path('models/fraud_pipeline.joblib'))
    test_score = pipeline.predict_proba(x_test)[:, list(pipeline.classes_).index(1)]
    results = {
        'selection_metric': config['selection_metric'], 'threshold_metric': config['threshold_metric'],
        'selected_model': selected, 'threshold': threshold, 'dataset_sha256': audit['sha256'],
        'seed': config['seed'], 'cv_folds': config['cv_folds'],
        'candidate_parameters': {name: model.get_params() for name, (model, _) in candidates(config['seed']).items()},
        'numeric_features': config['numeric_features'], 'categorical_features': config['categorical_features'],
        'split_manifest': 'reports/metrics/data_audit.json',
        'threshold_grid': {'minimum': 0.05, 'maximum': 0.95, 'step': 0.01},
        'training_rows': len(x_train), 'validation_rows': len(x_valid), 'test_rows': len(x_test),
        'cv_comparison': comparisons, 'final_fit_seconds': round(fit_seconds, 3),
        'validation_selected_threshold': assess(y_valid, validation_score, threshold),
        'validation_threshold_0_5': assess(y_valid, validation_score, 0.5),
        'test_selected_threshold': assess(y_test, test_score, threshold),
        'test_threshold_0_5': assess(y_test, test_score, 0.5),
        'versions': {'python': platform.python_version(), **{name: metadata.version(name) for name in ('numpy','pandas','scikit-learn','joblib')}},
    }
    path('reports/metrics/model_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps({'selected_model': selected, 'threshold': threshold, 'validation': results['validation_selected_threshold'], 'test': results['test_selected_threshold']}, indent=2))
    return results


if __name__ == '__main__':
    train()
