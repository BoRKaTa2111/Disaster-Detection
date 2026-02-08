import runpy
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt


class TestData:
    class_names = ["Fire_Damage", "Land_Disaster", "Water_Disaster", "Non_Damage"]

    def cache(self): return self

    def shuffle(self, *a, **k): return self

    def prefetch(self, *a, **k): return self



class MockHist:
    history = {
        "accuracy": [0.3, 0.4],
        "val_accuracy": [0.28, 0.38],
        "loss": [1.2, 1.0],
        "val_loss": [1.3, 1.1],
    }

def cache(self): return self
def shuffle(self, *a, **k): return self
def prefetch(self, *a, **k): return self


def test_training_script_runs_and_calls_expected_things(monkeypatch):
    calls = {
        "image_ds": [],
        "fit": [],
        "evaluate": [],
        "save": [],
        "show": 0,
    }

    def fake_image_dataset_from_directory(*args, **kwargs):
        calls["image_ds"].append((args, kwargs))
        return TestData()

    monkeypatch.setattr(tf.keras.utils, "image_dataset_from_directory", fake_image_dataset_from_directory)

    def fake_fit(self, *args, **kwargs):
        calls["fit"].append((args, kwargs))
        return MockHist()

    def fake_evaluate(self, *args, **kwargs):
        calls["evaluate"].append((args, kwargs))
        return (0.0, 0.5)

    def fake_save(self, path, *args, **kwargs):
        calls["save"].append((path, args, kwargs))
        return None

    monkeypatch.setattr(models.Sequential, "fit", fake_fit, raising=True)
    monkeypatch.setattr(models.Sequential, "evaluate", fake_evaluate, raising=True)
    monkeypatch.setattr(models.Sequential, "save", fake_save, raising=True)

    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.__setitem__("show", calls["show"] + 1))

    runpy.run_path("data/train/model.py")
    assert len(calls["image_ds"]) >= 1
    assert len(calls["fit"]) == 1
    _, fit_kwargs = calls["fit"][0]
    assert "epochs" in fit_kwargs
    assert fit_kwargs["epochs"] > 0
    assert len(calls["evaluate"]) == 1
    assert len(calls["save"]) == 1
    saved_path, _, _ = calls["save"][0]
    assert isinstance(saved_path, str) and saved_path
    assert calls["show"] >= 0
