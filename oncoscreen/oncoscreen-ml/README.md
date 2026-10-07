# OncoScreen ML

Train a breast tumor classifier (benign vs malignant) on the Wisconsin Diagnostic dataset that ships with scikit-learn. No downloads needed.

> Educational project. Not a medical device.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py
pytest
```

## Output

- `model.joblib`: trained pipeline + feature names + class names
- `metrics.json`: ROC-AUC on the held-out test set

## Model

StandardScaler + RandomForest (300 trees), 80/20 stratified split, seed 42.

## Next

Serve the model with [oncoscreen-api](../oncoscreen-api). Copy `model.joblib` into that repo.

## Ideas to extend

- Try XGBoost / SVM and compare
- Add SHAP feature importance plots
- Cross-validation + hyperparameter search

## License

MIT
