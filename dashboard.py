"""Interactive inspection of previously generated experiment metrics."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title='ML Benchmark Lab', layout='wide')
st.title('ML Foundations to Production')
st.caption('Reproducible Bayesian, classical, deep learning, and unsupervised experiments')
path = st.text_input('Artifacts directory', 'artifacts')
metric_file = Path(path)/'metrics.json'
if not metric_file.is_file():
    st.info('Run `ml-lab --task all` in your terminal first to generate metrics.')
    st.stop()
data = json.loads(metric_file.read_text())
tab1, tab2, tab3 = st.tabs(['Regression', 'Classification', 'Clustering'])
with tab1:
    rows = {name: vals for name, vals in data['regression'].items() if isinstance(vals, dict) and 'rmse' in vals}
    st.dataframe(pd.DataFrame(rows).T)
    st.write('Bayesian degree selected by BIC:', data['regression']['selected_degree'])
    st.line_chart(pd.Series(data['regression']['bic_by_degree'], name='BIC'))
with tab2:
    rows = {name: vals for name, vals in data['classification'].items() if isinstance(vals, dict) and 'accuracy' in vals}
    st.dataframe(pd.DataFrame(rows).T)
    st.subheader('Fixed-classifier Hoeffding bound on held-out examples')
    st.json(data['classification']['svm_hoeffding'])
with tab3:
    st.json(data['clustering'])
    st.caption('Adjusted Rand Index compares unsupervised clusters against labels only for evaluation.')
