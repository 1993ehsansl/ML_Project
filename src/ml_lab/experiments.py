"""End-to-end benchmarks with held-out evaluation and persistent artifacts."""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, adjusted_rand_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from .data import regression_data, digits_data
from .bayesian import BayesianPolynomialRegression, select_degree
from .clustering import GaussianMixtureEM
from .neural import RegressionMLP, DigitsCNN, train_torch, torch_predict
from .uncertainty import empirical_risk_bound


def _save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def run_regression(out='artifacts', seed=42, quick=False):
    import torch
    path = Path(out); path.mkdir(parents=True, exist_ok=True)
    xtr, xte, ytr, yte = regression_data(n=160 if quick else 400, seed=seed)
    # Degree selection happens ONLY on train split.
    degree, bics = select_degree(xtr, ytr, degrees=range(1, 7))
    bayes = BayesianPolynomialRegression(degree).fit(xtr, ytr)
    svr = make_pipeline(StandardScaler(), SVR(C=10, gamma='scale')).fit(xtr, ytr)
    torch.manual_seed(seed)
    mlp, history = train_torch(RegressionMLP(), xtr, ytr, task='regression',
                               epochs=5 if quick else 120, seed=seed)
    models = {'bayesian': bayes.predict(xte), 'svr': svr.predict(xte),
              'mlp': torch_predict(mlp, xte, 'regression')}
    metrics = {name: {'rmse': float(np.sqrt(mean_squared_error(yte, pred))),
                      'r2': float(r2_score(yte, pred))} for name, pred in models.items()}
    mean, std = bayes.predict(xte, return_std=True)
    metrics['bayesian']['interval_95_coverage'] = float(np.mean(np.abs(yte-mean) <= 1.96*std))
    metrics['selected_degree'] = degree
    metrics['bic_by_degree'] = bics
    metrics['mlp_final_train_loss'] = history[-1]
    joblib.dump(bayes, path/'bayesian.joblib')
    joblib.dump(svr, path/'svr.joblib')
    torch.save(mlp.state_dict(), path/'regression_mlp.pt')
    _save_json(path/'regression_metrics.json', metrics)
    return metrics


def run_classification(out='artifacts', seed=42, quick=False):
    import torch
    path = Path(out); path.mkdir(parents=True, exist_ok=True)
    Xtr, Xte, ytr, yte = digits_data(seed)
    models = {
        'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500)),
        'svm': make_pipeline(StandardScaler(), SVC(C=10, kernel='rbf', probability=True, random_state=seed))
    }
    metrics = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        metrics[name] = {'accuracy': float(accuracy_score(yte, pred)),
                         'macro_f1': float(f1_score(yte, pred, average='macro'))}
        joblib.dump(model, path/f'{name}.joblib')
    torch.manual_seed(seed)
    # sklearn digits pixels are 0..16, converted to 0..1 for CNN
    cnn, history = train_torch(DigitsCNN(), (Xtr/16.).reshape(-1, 1, 8, 8), ytr,
                               epochs=2 if quick else 25, seed=seed)
    pred = torch_predict(cnn, (Xte/16.).reshape(-1, 1, 8, 8))
    metrics['cnn'] = {'accuracy': float(accuracy_score(yte, pred)),
                      'macro_f1': float(f1_score(yte, pred, average='macro')),
                      'train_loss_last': history[-1]}
    metrics['svm_hoeffding'] = empirical_risk_bound(models['svm'].predict(Xte) != yte)
    metrics['evaluated_samples'] = len(yte)
    torch.save(cnn.state_dict(), path/'cnn.pt')
    _save_json(path/'classification_metrics.json', metrics)
    return metrics


def run_clustering(out='artifacts', seed=42, quick=False):
    path = Path(out); path.mkdir(parents=True, exist_ok=True)
    Xtr, Xte, ytr, yte = digits_data(seed)
    # Fit preprocessing only to training data, and compare against sklearn EM.
    scaler = StandardScaler().fit(Xtr)
    pca = PCA(n_components=2, random_state=seed).fit(scaler.transform(Xtr))
    X = pca.transform(scaler.transform(Xte))
    scratch = GaussianMixtureEM(10, max_iter=10 if quick else 100, seed=seed).fit(X)
    reference = GaussianMixture(n_components=10, reg_covar=1e-5, random_state=seed).fit(X)
    metrics = {'custom_em_ari': float(adjusted_rand_score(yte, scratch.predict(X))),
               'sklearn_em_ari': float(adjusted_rand_score(yte, reference.predict(X))),
               'custom_em_iterations': scratch.n_iter_,
               'custom_em_last_log_likelihood': scratch.history_[-1]}
    joblib.dump({'scaler': scaler, 'pca': pca, 'gmm': scratch}, path/'clustering.joblib')
    _save_json(path/'clustering_metrics.json', metrics)
    return metrics


def run_all(out='artifacts', seed=42, quick=False, track=False):
    results = {'regression': run_regression(out, seed, quick),
               'classification': run_classification(out, seed, quick),
               'clustering': run_clustering(out, seed, quick)}
    _save_json(Path(out)/'metrics.json', results)
    if track:
        import mlflow
        with mlflow.start_run(run_name='ml-lab-all'):
            mlflow.log_params({'seed': seed, 'quick': quick})
            for group, values in results.items():
                for name, value in values.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f'{group}.{name}', float(value))
            mlflow.log_artifact(str(Path(out)/'metrics.json'))
    return results
