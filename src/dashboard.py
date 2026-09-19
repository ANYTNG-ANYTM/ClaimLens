"""Streamlit client for a completed claim review record."""
import json
import os
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui_client import request_prediction
from src.util import path


def main():
    st.title('Insurance claim review assistance')
    st.caption('A model flag is a prompt for human review, not proof of fraud. Use after initial damage assessment.')
    example_path = path('reports/example_request.json')
    example = example_path.read_text(encoding='utf-8') if example_path.exists() else '{}'
    with st.form('claim'):
        claim_text = st.text_area('Claim fields as JSON', value=example, height=350)
        submitted = st.form_submit_button('Score claim')
    if submitted:
        try:
            claim = json.loads(claim_text)
            if not isinstance(claim, dict):
                raise ValueError('Enter one JSON object')
            result = request_prediction(claim, os.getenv('FRAUD_API_URL', 'http://localhost:8080'))
            st.metric('Fraud review score', f"{result['fraud_score']:.3f}")
            st.write(f"Decision threshold: {result['threshold']:.2f}")
            if result['review_recommended']:
                st.warning('Flagged for human fraud review')
            else:
                st.success('No model flag; standard review still applies')
            st.caption('Score is an uncalibrated model output, not a proven fraud probability.')
        except Exception as exc:
            st.error(f'Prediction failed: {exc}')


if __name__ == '__main__':
    main()
