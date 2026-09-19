"""Build a synthetic demonstration claim from development medians and modes."""
import json
from src.data_pipeline import prepare
from src.util import load_config, path


def main():
    config = load_config()
    parts, _ = prepare(save=False)
    train = parts['train']
    example = {name: float(train[name].median()) for name in config['numeric_features']}
    example.update({name: str(train[name].loc[train[name].ne('?')].mode().iloc[0]) for name in config['categorical_features']})
    output = path('reports/example_request.json')
    output.write_text(json.dumps(example, indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
