"""
tests/test_models.py

Smoke tests for the model factory across XGBoost, LightGBM, CatBoost.
"""

from __future__ import annotations

import numpy as np
import pytest

from bad_channel_rejection.models import SUPPORTED_MODELS, create_model


@pytest.fixture(scope="module")
def small_binary_data(feature_matrix, preprocessed_X):
    X = preprocessed_X["X"]
    y = feature_matrix["y_hard"]
    np.random.seed(0)
    pos_idx = np.where(y == 1)[0][:400]
    neg_idx = np.where(y == 0)[0][:3000]
    idx = np.concatenate([pos_idx, neg_idx])
    np.random.shuffle(idx)
    X_sub, y_sub = X[idx], y[idx]
    split = int(0.8 * len(y_sub))
    return (X_sub[:split], y_sub[:split]), (X_sub[split:], y_sub[split:])


@pytest.mark.parametrize("model_name", list(SUPPORTED_MODELS))
def test_model_fit_predict(small_binary_data, model_name):
    (X_tr, y_tr), (X_va, y_va) = small_binary_data
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = create_model(model_name, scale_pos_weight=spw)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))

    probs = model.predict_proba(X_va)
    assert probs.shape == (len(y_va), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert model.best_iteration >= 1


@pytest.mark.parametrize("model_name", list(SUPPORTED_MODELS))
def test_model_with_sample_weights(small_binary_data, model_name):
    (X_tr, y_tr), (X_va, y_va) = small_binary_data
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    w_tr = np.random.RandomState(0).uniform(0.1, 1.0, size=len(y_tr))

    model = create_model(model_name, scale_pos_weight=spw)
    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_va, y_va))
    probs = model.predict_proba(X_va)
    assert probs.shape == (len(y_va), 2)


@pytest.mark.parametrize("model_name", list(SUPPORTED_MODELS))
def test_model_save_load_roundtrip(small_binary_data, model_name, tmp_path):
    (X_tr, y_tr), (X_va, _) = small_binary_data
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    from bad_channel_rejection.models import MODEL_EXT

    model = create_model(model_name, scale_pos_weight=spw)
    model.fit(X_tr, y_tr)

    path = tmp_path / f"model.{MODEL_EXT[model_name]}"
    model.save(path)
    assert path.exists()

    loaded = create_model(model_name, scale_pos_weight=spw).load(path)
    probs_orig = model.predict_proba(X_va)[:, 1]
    probs_loaded = loaded.predict_proba(X_va)[:, 1]
    assert np.allclose(probs_orig, probs_loaded, atol=1e-5)


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        create_model("randomforest", scale_pos_weight=1.0)
