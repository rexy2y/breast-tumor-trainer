"""Train a breast-tumor classifier and save it for the API."""
import json

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline():
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=300, random_state=42),
    )


def main():
    data = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=0.2, stratify=data.target, random_state=42
    )
    model = build_pipeline().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])

    print(classification_report(y_te, pred, target_names=data.target_names))
    print(f"ROC-AUC: {auc:.4f}")

    joblib.dump(
        {
            "model": model,
            "features": list(data.feature_names),
            "classes": list(data.target_names),
        },
        "model.joblib",
    )
    with open("metrics.json", "w") as f:
        json.dump({"roc_auc": round(auc, 4)}, f)


if __name__ == "__main__":
    main()
