"""Unit tests for the registry promotion gate (no MLflow server needed: a fake client)."""
import sys
from pathlib import Path

import pytest
from mlflow.exceptions import MlflowException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tracking  # noqa: E402


class _Version:
    def __init__(self, version, tags):
        self.version, self.tags = str(version), dict(tags)


class FakeClient:
    """Just enough of MlflowClient for register_and_promote."""

    def __init__(self):
        self.tags = {}      # version -> tags
        self.aliases = {}   # alias -> version

    def set_model_version_tag(self, name, version, key, value):
        self.tags.setdefault(str(version), {})[key] = value

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = str(version)

    def get_model_version_by_alias(self, name, alias):
        if alias not in self.aliases:
            raise MlflowException(f"alias {alias} not found")
        v = self.aliases[alias]
        return _Version(v, self.tags.get(v, {}))


def promote(client, version, val_loss, fp="fp1", verified=True, **kw):
    return tracking.register_and_promote(client, "m", version, "run", val_loss, fp, verified, "ok",
                                         log=lambda *_: None, **kw)


def test_first_version_becomes_champion():
    c = FakeClient()
    promoted, msg = promote(c, 1, 0.50)
    assert promoted and c.aliases["champion"] == "1" and c.aliases["challenger"] == "1"
    assert "first version" in msg


def test_identical_retrain_does_not_replace_champion():
    c = FakeClient()
    promote(c, 1, 0.09604676067829132)
    promoted, msg = promote(c, 2, 0.09604676067829132)  # exactly the same loss, unrounded
    assert not promoted and c.aliases["champion"] == "1" and c.aliases["challenger"] == "2"
    assert "does not beat" in msg


def test_improvement_below_margin_is_noise():
    c = FakeClient()
    promote(c, 1, 0.1000)
    promoted, _ = promote(c, 2, 0.09995)  # 0.05% better < 0.1% margin
    assert not promoted and c.aliases["champion"] == "1"


def test_clear_improvement_is_promoted():
    c = FakeClient()
    promote(c, 1, 0.1000)
    promoted, msg = promote(c, 2, 0.0900)
    assert promoted and c.aliases["champion"] == "2" and "beats champion" in msg


def test_worse_model_not_promoted():
    c = FakeClient()
    promote(c, 1, 0.1000)
    promoted, _ = promote(c, 2, 0.2000)
    assert not promoted and c.aliases["champion"] == "1"


def test_margin_is_configurable():
    c = FakeClient()
    promote(c, 1, 0.1000)
    promoted, _ = promote(c, 2, 0.0999, margin=0.0)  # tiny but real improvement, no margin required
    assert promoted


def test_different_data_fingerprint_blocks_auto_promotion():
    c = FakeClient()
    promote(c, 1, 0.1000, fp="A")
    promoted, msg = promote(c, 2, 0.0100, fp="B")
    assert not promoted and "fingerprint" in msg and c.aliases["champion"] == "1"


def test_failed_verification_blocks_even_the_first_version():
    c = FakeClient()
    promoted, msg = promote(c, 1, 0.1, verified=False)
    assert not promoted and "verification failed" in msg and "champion" not in c.aliases


def test_acceptance_target_blocks_promotion():
    c = FakeClient()
    promoted, msg = promote(c, 1, 0.1, target_ok=False, target_msg="test AUC 0.9 vs required >= 0.99")
    assert not promoted and "acceptance target not met" in msg and "champion" not in c.aliases


# ----------------------------- acceptance targets ----------------------------- #
CLS = {"auc_macro_ovr": 0.995, "auc_ci95_low": 0.993, "auc_ci95_high": 0.997}
REG = {"mse": 0.25, "mse_se": 0.01}


def test_targets_classification():
    assert tracking.check_targets("classification", CLS, min_auc=0.99)[0]
    ok, msg = tracking.check_targets("classification", CLS, min_auc=0.999)
    assert not ok and "0.9950" in msg


def test_targets_regression():
    assert tracking.check_targets("regression", REG, max_mse=0.30)[0]
    ok, msg = tracking.check_targets("regression", REG, max_mse=0.20)
    assert not ok and "0.2500" in msg


def test_no_target_is_ok():
    ok, msg = tracking.check_targets("regression", REG)
    assert ok and "no acceptance target" in msg
