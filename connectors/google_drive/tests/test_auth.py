from unittest.mock import MagicMock

import pytest

import auth


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "LONGBRAIN_HOME", tmp_path)
    monkeypatch.setattr(auth, "TOKEN_FILE", tmp_path / "google_drive_token.json")
    monkeypatch.setattr(auth, "CLIENT_SECRET_FILE", tmp_path / "google_oauth_client.json")
    return tmp_path


def test_raises_clear_error_when_client_secret_missing():
    # No cached token and no client secret -> can't start the OAuth flow.
    with pytest.raises(FileNotFoundError, match="google_oauth_client.json"):
        auth.get_credentials()


def test_reuses_valid_cached_token(monkeypatch):
    auth.TOKEN_FILE.write_text('{"fake": "token"}')
    fake_creds = MagicMock(valid=True)
    monkeypatch.setattr(
        auth.Credentials, "from_authorized_user_file",
        classmethod(lambda cls, path, scopes: fake_creds),
    )

    result = auth.get_credentials()

    assert result is fake_creds


def test_refreshes_expired_token_without_reauthorizing(monkeypatch, isolated_home):
    auth.TOKEN_FILE.write_text('{"fake": "token"}')
    fake_creds = MagicMock(valid=False, expired=True, refresh_token="r1")
    fake_creds.to_json.return_value = '{"fake": "refreshed_token"}'
    monkeypatch.setattr(
        auth.Credentials, "from_authorized_user_file",
        classmethod(lambda cls, path, scopes: fake_creds),
    )
    flow_mock = MagicMock()
    monkeypatch.setattr(auth.InstalledAppFlow, "from_client_secrets_file", flow_mock)

    result = auth.get_credentials()

    assert result is fake_creds
    fake_creds.refresh.assert_called_once()
    flow_mock.assert_not_called()  # must not open a browser when refresh works
    assert auth.TOKEN_FILE.read_text() == fake_creds.to_json()
