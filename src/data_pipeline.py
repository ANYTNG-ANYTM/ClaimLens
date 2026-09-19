"""Raw data validation and fixed, disjoint development splits."""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .util import load_config, path


def read_data(config=None):
    config = config or load_config()
    csv_path = path(config['dataset'])
    if not csv_path.is_file():
        raise FileNotFoundError(f'Missing private dataset at {csv_path}. Supply a compatible licensed CSV at the configured path.')
    data = pd.read_csv(csv_path)
    needed = set(config['numeric_features'] + config['categorical_features'] + [config['target'], 'policy_number'])
    missing = needed.difference(data.columns)
    if missing:
        raise ValueError(f'Dataset lacks columns: {sorted(missing)}')
    if data[config['target']].isna().any() or set(data[config['target']]) != {'Y', 'N'}:
        raise ValueError('Expected fraud_reported labels Y and N without missing values')
    if data.policy_number.isna().any() or data.policy_number.duplicated().any():
        raise ValueError('Policy identifiers must be unique and nonmissing')
    if data.duplicated().any():
        raise ValueError('Duplicate rows require review before splitting')
    return data


def dataset_sha256(config=None):
    config = config or load_config()
    return hashlib.sha256(path(config['dataset']).read_bytes()).hexdigest()


def split_data(data, config=None):
    config = config or load_config()
    idx = np.arange(len(data))
    train_valid, test = train_test_split(idx, test_size=config['test_size'], random_state=config['seed'], stratify=data[config['target']])
    validation_fraction = config['validation_size'] / (1 - config['test_size'])
    train, valid = train_test_split(train_valid, test_size=validation_fraction, random_state=config['seed'], stratify=data.iloc[train_valid][config['target']])
    parts = {'train': data.iloc[train].copy(), 'validation': data.iloc[valid].copy(), 'test': data.iloc[test].copy()}
    ids = [set(part.policy_number) for part in parts.values()]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Policy overlap across splits')
    return parts


def prepare(save=True):
    config = load_config()
    data = read_data(config)
    parts = split_data(data, config)
    quality = {
        'rows': len(data), 'columns': len(data.columns), 'sha256': dataset_sha256(config),
        'labels': data[config['target']].value_counts().to_dict(),
        'missing_by_column': data.isna().sum().loc[lambda x: x > 0].to_dict(),
        'question_mark_by_column': data.astype(str).eq('?').sum().loc[lambda x: x > 0].to_dict(),
        'constant_columns': data.columns[data.nunique(dropna=False) == 1].tolist(),
        'duplicate_rows': int(data.duplicated().sum()),
        'unique_policies': int(data.policy_number.nunique()),
        'bad_date_order': int((pd.to_datetime(data.policy_bind_date) > pd.to_datetime(data.incident_date)).sum()),
        'claim_sum_mismatch': int((data.total_claim_amount != data[['injury_claim','property_claim','vehicle_claim']].sum(axis=1)).sum()),
        'numeric_ranges': {column: [float(data[column].min()), float(data[column].max())] for column in config['numeric_features']},
        'categorical_cardinalities': {column: int(data[column].nunique(dropna=False)) for column in config['categorical_features']},
        'incident_date_range': [str(data.incident_date.min()), str(data.incident_date.max())],
        'policy_bind_date_range': [str(data.policy_bind_date.min()), str(data.policy_bind_date.max())],
        'splits': {name: {'rows': len(part), 'fraud': int(part[config['target']].eq('Y').sum()), 'policy_numbers': part.policy_number.astype(int).tolist()} for name, part in parts.items()},
    }
    if save:
        path('data/processed').mkdir(parents=True, exist_ok=True)
        path('reports/metrics').mkdir(parents=True, exist_ok=True)
        for name, part in parts.items():
            part.to_csv(path(f'data/processed/{name}.csv'), index=False)
        path('reports/metrics/data_audit.json').write_text(json.dumps(quality, indent=2), encoding='utf-8')
    return parts, quality


if __name__ == '__main__':
    _, result = prepare()
    print(json.dumps({key: result[key] for key in ('rows','columns','sha256','labels','duplicate_rows','unique_policies','bad_date_order')}, indent=2))
