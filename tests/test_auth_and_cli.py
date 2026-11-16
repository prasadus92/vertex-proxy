from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from google.auth import exceptions as google_exceptions

from vertex_proxy import __main__ as cli
from vertex_proxy import auth
from vertex_proxy.auth import CredentialsError, TokenManager, load_credentials


class _Credentials:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.token: str | None = None
        self.expiry = None
        self.valid = False

    def refresh(self, _request: Any) -> None:
        if self.error is not None:
            raise self.error
        self.token, self.valid = "t", True


def test_missing_credentials_file(tmp_path: Path) -> None:
    with pytest.raises(CredentialsError, match="credentials file not found"):
        load_credentials(tmp_path / "missing.json")


def test_unreadable_credentials_file(tmp_path: Path) -> None:
    key = tmp_path / "key.json"
    key.write_text("not json")
    with pytest.raises(CredentialsError, match="cannot read credentials file"):
        load_credentials(key)


def test_unknown_credentials_type(tmp_path: Path) -> None:
    key = tmp_path / "key.json"
    key.write_text('{"type": "something_else"}')
    with pytest.raises(CredentialsError, match="credential type 'something_else'"):
        load_credentials(key)


def test_service_account_key_is_loaded_with_its_project(tmp_path: Path) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    pem = rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    key = tmp_path / "key.json"
    key.write_text(
        json.dumps(
            {
                "type": "service_account",
                "project_id": "from-key",
                "private_key_id": "1",
                "private_key": pem.decode(),
                "client_email": "sa@from-key.iam.gserviceaccount.com",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        )
    )
    _, project = load_credentials(key)
    assert project == "from-key"


def test_no_application_default_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_adc(**_kwargs: Any) -> Any:
        raise google_exceptions.DefaultCredentialsError("nothing found")

    monkeypatch.setattr(auth.google.auth, "default", no_adc)
    with pytest.raises(CredentialsError, match="gcloud auth application-default login"):
        load_credentials(None)


async def test_rejected_refresh_is_explained() -> None:
    creds = _Credentials(google_exceptions.RefreshError("invalid_grant: account disabled"))
    manager = TokenManager(creds, "p")  # type: ignore[arg-type]
    with pytest.raises(CredentialsError, match="Google rejected the credentials"):
        await manager.get_token()


async def test_token_is_refreshed_only_when_invalid() -> None:
    creds = _Credentials()
    manager = TokenManager(creds, "p")  # type: ignore[arg-type]
    assert await manager.get_token() == "t"
    creds.error = RuntimeError("must not refresh a valid token")
    assert await manager.get_token() == "t"


def test_cli_reports_bad_credentials_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["--credentials", str(tmp_path / "missing.json"), "--project-id", "p"])
    assert code == 2
    assert capsys.readouterr().err.startswith("vertex-proxy: credentials file not found")


def test_cli_reports_a_missing_config_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--config", str(tmp_path / "nope.toml")]) == 2
    assert "config file not found" in capsys.readouterr().err


def test_cli_requires_a_project(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli.TokenManager, "from_file", lambda *_a: TokenManager(_Credentials(), None)
    )  # type: ignore[arg-type]
    assert cli.main([]) == 2
    assert "no Google Cloud project" in capsys.readouterr().err


_POOL = "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/x"
_JWT = "urn:ietf:params:oauth:token-type:jwt"
_STS = "https://sts.googleapis.com/v1/token"


@pytest.mark.parametrize(
    ("info", "module"),
    [
        (
            {
                "type": "external_account",
                "audience": _POOL,
                "subject_token_type": _JWT,
                "token_url": _STS,
                "credential_source": {"file": "/var/run/token"},
            },
            "google.auth.identity_pool",
        ),
        (
            {
                "type": "external_account",
                "audience": _POOL,
                "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
                "token_url": _STS,
                "credential_source": {
                    "environment_id": "aws1",
                    "region_url": "http://169.254.169.254/latest/meta-data/placement/availability-zone",
                    "url": "http://169.254.169.254/latest/meta-data/iam/security-credentials",
                    "regional_cred_verification_url": "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15",
                },
            },
            "google.auth.aws",
        ),
        (
            {
                "type": "external_account",
                "audience": _POOL,
                "subject_token_type": _JWT,
                "token_url": _STS,
                "credential_source": {"executable": {"command": "/usr/bin/get-token"}},
            },
            "google.auth.pluggable",
        ),
        (
            {
                "type": "external_account_authorized_user",
                "audience": "//iam.googleapis.com/locations/global/workforcePools/p/providers/x",
                "refresh_token": "r",
                "token_url": "https://sts.googleapis.com/v1/oauthtoken",
                "token_info_url": "https://sts.googleapis.com/v1/introspect",
                "client_id": "c",
                "client_secret": "s",
            },
            "google.auth.external_account_authorized_user",
        ),
        (
            {
                "type": "impersonated_service_account",
                "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com:generateAccessToken",
                "source_credentials": {
                    "type": "authorized_user",
                    "client_id": "c",
                    "client_secret": "s",
                    "refresh_token": "r",
                },
            },
            "google.auth.impersonated_credentials",
        ),
        (
            {
                "type": "authorized_user",
                "client_id": "c",
                "client_secret": "s",
                "refresh_token": "r",
            },
            "google.oauth2.credentials",
        ),
    ],
)
def test_every_supported_credential_type_loads(
    tmp_path: Path, info: dict[str, Any], module: str
) -> None:
    key = tmp_path / "creds.json"
    key.write_text(json.dumps(info))
    creds, _ = load_credentials(key)
    assert type(creds).__module__ == module


async def test_concurrent_requests_refresh_the_token_once() -> None:
    import asyncio

    class Counting(_Credentials):
        calls = 0

        def refresh(self, request: Any) -> None:
            Counting.calls += 1
            time.sleep(0.05)
            super().refresh(request)

    manager = TokenManager(Counting(), "p")  # type: ignore[arg-type]
    await asyncio.gather(*(manager.get_token() for _ in range(5)))
    assert Counting.calls == 1
