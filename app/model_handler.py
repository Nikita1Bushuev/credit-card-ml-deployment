import numpy as np
import joblib
import os

model_dir= 'models'
model_files = {
    'v1': 'model_v1.pkl',
    'v2': 'model_v2.pkl'
}

_models = {}
def get_model(version: str = 'v1'):
    if version not in model_files:
        raise ValueError(f'Unknown model version: {version}')
    if version not in _models:
        path = os.path.join(model_dir, model_files[version])
        _models[version] = joblib.load(path)
    return _models[version]

def predict(features: np.ndarray, version: str = 'v1'):
    model = get_model(version)
    prediction  = model.predict(features)[0]
    probability  = model.predict_proba(features)[0, 1]
    return int(prediction), float(probability)