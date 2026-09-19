"""Fold-local transformations for raw claim records."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def features(frame, config):
    columns = config['numeric_features'] + config['categorical_features']
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f'Missing required features: {sorted(missing)}')
    result = frame[columns].copy()
    for column in config['numeric_features']:
        source = result[column].replace(['?', ''], np.nan)
        converted = pd.to_numeric(source, errors='coerce')
        if (source.notna() & converted.isna()).any():
            raise ValueError(f'Invalid numeric value for {column}')
        if np.isinf(converted.to_numpy(dtype=float)).any():
            raise ValueError(f'Non-finite numeric value for {column}')
        result[column] = converted
    for column in config['categorical_features']:
        result[column] = result[column].replace(['?', ''], np.nan).astype('object')
        result[column] = result[column].where(result[column].isna(), result[column].astype(str))
    return result


def make_preprocessor(config, scale=False):
    numeric = [('imputer', SimpleImputer(strategy='median'))]
    if scale:
        numeric.append(('scaler', StandardScaler()))
    categorical = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('numeric', Pipeline(numeric), config['numeric_features']),
        ('categorical', categorical, config['categorical_features']),
    ], verbose_feature_names_out=True)
