"""Development-only EDA. Structural quality covers all rows; associations use development rows."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data_pipeline import prepare
from .util import path


def run():
    parts, audit = prepare()
    dev = pd.concat([parts['train'], parts['validation']], ignore_index=True)
    figures = path('reports/figures')
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid')
    dev.fraud_reported.value_counts().reindex(['N','Y']).plot.bar(color=['#577590','#d35f50'])
    plt.title('Development labels (test excluded)')
    plt.ylabel('Claims')
    plt.tight_layout(); plt.savefig(figures/'target_distribution.png'); plt.close()
    missing = {key: value for key, value in audit['question_mark_by_column'].items() if value}
    pd.Series(missing).sort_values().plot.barh(color='#577590')
    plt.title('Question mark placeholders in full CSV')
    plt.tight_layout(); plt.savefig(figures/'missingness.png'); plt.close()
    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    for axis, column in zip(axes, ['injury_claim','property_claim','vehicle_claim']):
        sns.boxplot(data=dev, x='fraud_reported', y=column, ax=axis, showfliers=False)
        axis.set_title(column)
    fig.suptitle('Development claim amounts by label')
    fig.tight_layout(); fig.savefig(figures/'claim_amounts.png'); plt.close(fig)
    rates = dev.groupby('incident_severity').fraud_reported.agg(count='size', fraud_rate=lambda s: s.eq('Y').mean()).sort_values('fraud_rate')
    rates.fraud_rate.plot.barh(color='#d35f50')
    plt.xlabel('Observed fraud label fraction'); plt.title('Development severity association')
    plt.tight_layout(); plt.savefig(figures/'severity_association.png'); plt.close()
    hobbies = dev.groupby('insured_hobbies').fraud_reported.agg(count='size', fraud_rate=lambda s: s.eq('Y').mean()).sort_values('fraud_rate')
    hobbies.fraud_rate.plot.barh(figsize=(7,7), color='#577590')
    plt.xlabel('Observed fraud label fraction'); plt.title('Development hobby association')
    plt.tight_layout(); plt.savefig(figures/'hobby_association.png'); plt.close()
    summary = {'development_rows': len(dev), 'development_fraud': int(dev.fraud_reported.eq('Y').sum()),
               'severity': rates.reset_index().to_dict(orient='records'), 'hobbies': hobbies.reset_index().to_dict(orient='records'),
               'full_data_quality': {key: audit[key] for key in ['rows','columns','duplicate_rows','constant_columns','question_mark_by_column','bad_date_order','claim_sum_mismatch']},
               'notes': ['Target associations use development rows only.', 'Policy number is unique and excluded from modelling.',
                         'Total claim equals the sum of component claims in every row and is excluded.',
                         'Severity has a strong label association and requires availability at intended review time.',
                         'The CSV has no verified provenance, sampling design, or label audit. Associations are not causal.']}
    path('reports/eda_findings.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'development_rows': len(dev), 'severity': summary['severity']}, indent=2))
    return summary


if __name__ == '__main__':
    run()
