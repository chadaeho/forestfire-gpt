"""
산불 발화위험 예측 모델 학습 스크립트
실제 산악기상관측망 데이터 연동 전, 더미 데이터로 모델 동작 확인용
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import pickle
import os

np.random.seed(42)

def generate_dummy_data(n=5000):
    """과거 산불이력을 모사한 더미 데이터 생성"""
    df = pd.DataFrame({
        'temperature': np.random.uniform(-5, 35, n),      # 기온(℃)
        'humidity': np.random.uniform(10, 95, n),         # 습도(%)
        'wind_speed': np.random.uniform(0, 20, n),        # 풍속(m/s)
        'fuel_moisture': np.random.uniform(5, 30, n),     # 연료습도(%)
        'ndvi': np.random.uniform(0.2, 0.9, n),           # 식생지수
        'slope': np.random.uniform(0, 45, n),             # 경사(도)
        'elevation': np.random.uniform(50, 1200, n),      # 고도(m)
    })
    # 발화 라벨: 고온·저습·강풍·저연료습도일수록 발화 확률↑
    logit = (
        0.08 * df['temperature']
        - 0.05 * df['humidity']
        + 0.15 * df['wind_speed']
        - 0.10 * df['fuel_moisture']
        + 0.02 * df['slope']
        - 1.5
    )
    prob = 1 / (1 + np.exp(-logit))
    df['fire'] = (np.random.rand(n) < prob).astype(int)
    return df


def train():
    df = generate_dummy_data()
    X = df.drop(columns=['fire'])
    y = df['fire']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        eval_metric='auc', random_state=42
    )
    model.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    print(f"✅ Validation AUC: {auc:.4f}")

    os.makedirs('models', exist_ok=True)
    with open('models/fire_risk.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Saved: models/fire_risk.pkl")
    return auc


if __name__ == '__main__':
    train()
