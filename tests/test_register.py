from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from mlflow.exceptions import MlflowException

from src import register as reg


class FakeClient:
    """Keeps aliases in a dict, like the registry does."""

    def __init__(self, aliases: dict[str, str] | None = None):
        self.aliases = dict(aliases or {})
        self.set_calls: list[tuple[str, str]] = []

    def get_model_version_by_alias(self, name, alias):
        if alias not in self.aliases:
            raise MlflowException(f"alias {alias} not found")
        return SimpleNamespace(version=self.aliases[alias])

    def set_registered_model_alias(self, name, alias, version):
        self.set_calls.append((alias, str(version)))
        self.aliases[alias] = str(version)

    def get_model_version(self, name, version):
        if int(version) > 99:
            raise MlflowException(f"version {version} not found")
        return SimpleNamespace(version=str(version))


def test_first_promotion_has_no_previous():
    client = FakeClient()
    assert reg.promote(client, "m", "1") == {"champion": "1", "previous": None}


def test_promotion_moves_old_champion_to_previous():
    client = FakeClient({"champion": "3", "previous": "2"})
    assert reg.promote(client, "m", "4") == {"champion": "4", "previous": "3"}
    assert client.set_calls == [("previous", "3"), ("champion", "4")]


def test_repromoting_same_version_keeps_previous():
    client = FakeClient({"champion": "4", "previous": "3"})
    assert reg.promote(client, "m", "4") == {"champion": "4", "previous": "3"}
    assert ("previous", "4") not in client.set_calls


def test_get_alias_version_missing_returns_none():
    assert reg.get_alias_version(FakeClient(), "m", "champion") is None


@pytest.fixture
def run():
    return SimpleNamespace(
        info=SimpleNamespace(run_id="abc123", run_name="bt-test"),
        data=SimpleNamespace(
            tags={"git_commit": "dummy"},
            metrics={"eval_f1": 0.75, "eval_accuracy": 0.7, "train_loss": 0.5},
        ),
    )


def test_version_tags(run):
    tags = reg.version_tags(run)
    assert tags["run_id"] == "abc123"
    assert tags["git_commit"] == "dummy"
    assert tags["dvc_raw_md5"] == "unknown"
    assert tags["eval_f1"] == "0.7500"
    assert "train_loss" not in tags


def test_register_tags_and_promotes(run, monkeypatch):
    client = MagicMock()
    client.get_run.return_value = run
    client.get_model_version_by_alias.side_effect = MlflowException("no alias")
    register_model = MagicMock(return_value=SimpleNamespace(version=5))
    monkeypatch.setattr(reg.mlflow, "register_model", register_model)

    result = reg.register("abc123", name="m", client=client)

    register_model.assert_called_once_with("runs:/abc123/model", "m")
    client.set_model_version_tag.assert_any_call("m", "5", "eval_f1", "0.7500")
    client.set_registered_model_alias.assert_called_once_with("m", "champion", "5")
    assert result["version"] == "5"
    assert result["champion"] == "5"


def test_register_no_promote_leaves_aliases(run, monkeypatch):
    client = MagicMock()
    client.get_run.return_value = run
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="2")
    monkeypatch.setattr(reg.mlflow, "register_model", MagicMock(return_value=SimpleNamespace(version=6)))

    result = reg.register("abc123", name="m", promote_to_champion=False, client=client)

    client.set_registered_model_alias.assert_not_called()
    assert result["version"] == "6"
    assert result["champion"] == "2"


def test_rollback_swaps_champion_and_previous():
    client = FakeClient({"champion": "5", "previous": "4"})
    result = reg.rollback(client, "m")
    assert result == {"model_name": "m", "rolled_back_from": "5", "champion": "4", "previous": "5"}
    # a second rollback undoes the first
    assert reg.rollback(client, "m")["champion"] == "5"


def test_rollback_to_given_version():
    client = FakeClient({"champion": "5", "previous": "4"})
    result = reg.rollback(client, "m", "2")
    assert result == {"model_name": "m", "rolled_back_from": "5", "champion": "2", "previous": "5"}


def test_rollback_without_previous_fails():
    with pytest.raises(SystemExit, match="no @previous"):
        reg.rollback(FakeClient({"champion": "1"}), "m")


def test_rollback_to_current_champion_fails():
    with pytest.raises(SystemExit, match="already @champion"):
        reg.rollback(FakeClient({"champion": "3", "previous": "2"}), "m", "3")


def test_rollback_to_missing_version_leaves_aliases():
    client = FakeClient({"champion": "3", "previous": "2"})
    with pytest.raises(MlflowException):
        reg.rollback(client, "m", "500")
    assert client.aliases == {"champion": "3", "previous": "2"}
