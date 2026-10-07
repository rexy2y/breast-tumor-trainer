from sklearn.datasets import load_breast_cancer

from train import build_pipeline


def test_pipeline_fits():
    d = load_breast_cancer()
    model = build_pipeline().fit(d.data, d.target)
    assert model.score(d.data, d.target) > 0.95
