"""Export aggregate-only results suitable for a public release copy."""
import json

from src.util import path


def main():
    results = json.loads(path('reports/metrics/model_results.json').read_text(encoding='utf-8'))
    public = {key: results[key] for key in (
        'selected_model', 'threshold', 'seed', 'cv_folds', 'training_rows',
        'validation_rows', 'test_rows', 'cv_comparison', 'final_fit_seconds',
        'validation_selected_threshold', 'validation_threshold_0_5',
        'test_selected_threshold', 'test_threshold_0_5', 'versions',
    )}
    public['test_status'] = 'Exploratory: an earlier test result was viewed before severity was added.'
    path('reports/results_summary.json').write_text(json.dumps(public, indent=2), encoding='utf-8')
    explanation = json.loads(path('reports/metrics/coefficients.json').read_text(encoding='utf-8'))
    path('reports/coefficients.json').write_text(json.dumps(explanation, indent=2), encoding='utf-8')
    print('Exported aggregate-only results and coefficient summary')


if __name__ == '__main__':
    main()
