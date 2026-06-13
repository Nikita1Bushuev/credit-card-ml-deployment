import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

data_path = os.path.join('data', 'UCI_Credit_Card.csv')
df = pd.read_csv(data_path)

X = df.drop(columns=['default.payment.next.month'])
y = df['default.payment.next.month']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_v2 = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=24, n_jobs=-1)
rf_v2.fit(X_train, y_train)

y_pred = rf_v2.predict(X_test)
f1 = f1_score(y_test, y_pred)
pr = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
print(f1, pr, rec)

model_path = os.path.join('models', 'model_v2.pkl')
joblib.dump(rf_v2, model_path)
print(f'Model v2 saved to {model_path}')