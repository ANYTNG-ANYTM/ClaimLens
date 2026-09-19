"""Create short notebooks that call the same source functions as the CLI."""
import nbformat as nb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks'
OUT.mkdir(exist_ok=True)


def write(name, cells):
    notebook = nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(c) if kind == 'md' else nb.v4.new_code_cell(c) for kind, c in cells])
    notebook.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.write(notebook, OUT / name)


setup = "from pathlib import Path\nimport sys\nroot = Path.cwd().resolve()\nif root.name == 'notebooks': root = root.parent\nsys.path.insert(0, str(root))"
write('01_data_validation.ipynb', [
    ('md', '# 1. Data validation and split\nThe CSV is read unchanged. The split is seeded and stratified by the label.'),
    ('code', setup), ('code', "from src.data_pipeline import prepare\nparts, audit = prepare()\n{k: (len(v), v.fraud_reported.eq('Y').sum()) for k, v in parts.items()}"),
    ('code', "{k: audit[k] for k in ['sha256', 'duplicate_rows', 'constant_columns', 'question_mark_by_column', 'bad_date_order']}")])
write('02_eda.ipynb', [
    ('md', '# 2. Development EDA\nTarget associations and plots use the development rows. Structural quality checks use the full CSV.'),
    ('code', setup), ('code', "from src.eda import run\nsummary = run()\nsummary['severity']")])
write('03_feature_selection.ipynb', [
    ('md', '# 3. Feature review\nFeatures are selected by documented availability at review time, before viewing test labels. Severity must be assessed before this model is used.'),
    ('code', setup), ('code', "from src.util import load_config\nfrom src.data_pipeline import prepare\nconfig = load_config()\nparts, _ = prepare(save=False)\nprint('Numeric:', config['numeric_features'])\nprint('Categorical:', config['categorical_features'])\nprint('Excluded:', sorted(set(parts['train'].columns) - set(config['numeric_features'] + config['categorical_features'] + [config['target']])))")])
write('04_preprocessing.ipynb', [
    ('md', '# 4. Fold-safe preprocessing\nThis demonstration fits on the training partition only. The model CLI handles fitting separately inside each CV fold.'),
    ('code', setup), ('code', "from src.data_pipeline import prepare\nfrom src.preprocessing import features, make_preprocessor\nfrom src.util import load_config\nconfig = load_config()\nparts, _ = prepare(save=False)\nprocessor = make_preprocessor(config, scale=True)\nx_train = features(parts['train'], config)\nx_valid = features(parts['validation'], config)\nprocessor.fit(x_train)\nprocessor.transform(x_train).shape, processor.transform(x_valid).shape")])
write('05_modelling.ipynb', [
    ('md', '# 5. Model evaluation\nRun `python -m src.modelling` to regenerate the bundle. This notebook reads the frozen report without changing the deployed model.'),
    ('code', setup), ('code', "import json\nfrom src.util import path\nresults = json.loads(path('reports/metrics/model_results.json').read_text())\nresults['cv_comparison']"),
    ('code', "{k: results[k] for k in ['selected_model','threshold','validation_selected_threshold','test_selected_threshold']}")])
