from __future__ import annotations

from pathlib import Path

import pytest

from vertex_proxy.config import ConfigError, Settings, load_settings
from vertex_proxy.vertex import VertexEndpoints, vertex_host


def test_defaults_use_the_global_endpoint() -> None:
    s = Settings(_env_file=None)
    assert (s.anthropic_region, s.gemini_region, s.maas_region) == ("global", "global", "global")
    assert (s.host, s.port, s.api_key) == ("127.0.0.1", 8787, None)


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERTEX_PROXY_ANTHROPIC_REGION", "us-east5")
    monkeypatch.setenv("VERTEX_PROXY_PORT", "9000")
    s = load_settings()
    assert (s.anthropic_region, s.port) == ("us-east5", 9000)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_toml_file_is_read(tmp_path: Path) -> None:
    config = _write(
        tmp_path / "config.toml",
        'project_id = "from-file"\nanthropic_region = "us"\n'
        '[anthropic_model_aliases]\nfast = "claude-haiku-5-5"\n',
    )
    s = load_settings(config)
    assert s.project_id == "from-file"
    assert s.anthropic_region == "us"
    assert s.anthropic_model_aliases == {"fast": "claude-haiku-5-5"}


def test_precedence_is_cli_then_env_then_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write(tmp_path / "c.toml", 'project_id = "file"\nport = 1111\nhost = "0.0.0.0"\n')
    monkeypatch.setenv("VERTEX_PROXY_PORT", "2222")
    monkeypatch.setenv("VERTEX_PROXY_PROJECT_ID", "env")
    s = load_settings(config, project_id="cli", host=None)
    assert (s.project_id, s.port, s.host) == ("cli", 2222, "0.0.0.0")


def test_config_path_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _write(tmp_path / "env.toml", 'project_id = "env-file"\n')
    monkeypatch.setenv("VERTEX_PROXY_CONFIG", str(config))
    assert load_settings().project_id == "env-file"


def test_default_config_path_is_used_when_present(tmp_path: Path) -> None:
    _write(tmp_path / "xdg" / "vertex-proxy" / "config.toml", 'project_id = "xdg"\n')
    assert load_settings().project_id == "xdg"


def test_missing_explicit_config_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_settings(tmp_path / "nope.toml")


def test_invalid_config_names_the_file(tmp_path: Path) -> None:
    config = _write(tmp_path / "bad.toml", 'port = "not-a-port"\n')
    with pytest.raises(ConfigError, match="bad.toml"):
        load_settings(config)


def test_credentials_path_expands_home() -> None:
    s = Settings(_env_file=None, credentials_path="~/key.json")
    assert s.credentials_path == Path.home() / "key.json"


@pytest.mark.parametrize(
    ("location", "host"),
    [
        ("global", "aiplatform.googleapis.com"),
        ("us", "aiplatform.us.rep.googleapis.com"),
        ("eu", "aiplatform.eu.rep.googleapis.com"),
        ("us-east5", "us-east5-aiplatform.googleapis.com"),
    ],
)
def test_vertex_host_per_location(location: str, host: str) -> None:
    assert vertex_host(location) == host


def test_endpoint_urls() -> None:
    e = VertexEndpoints("p")
    assert e.anthropic("global", "claude-opus-5-5", stream=True) == (
        "https://aiplatform.googleapis.com/v1/projects/p/locations/global"
        "/publishers/anthropic/models/claude-opus-5-5:streamRawPredict"
    )
    assert e.anthropic_count_tokens("us").endswith(
        "/locations/us/publishers/anthropic/models/count-tokens:rawPredict"
    )
    assert e.openai_chat("global") == (
        "https://aiplatform.googleapis.com/v1/projects/p/locations/global"
        "/endpoints/openapi/chat/completions"
    )


def test_example_config_file_is_valid() -> None:
    example = Path(__file__).parent.parent / "examples" / "config.toml"
    s = load_settings(example)
    assert s.project_id == "my-gcp-project"
    assert s.maas_model_aliases["deepseek"] == "deepseek-ai/deepseek-v4.1-flash-maas"


@pytest.mark.parametrize(
    "setting",
    [{"anthropic_region": "us-east5.evil.com"}, {"gemini_region": "a/b"}, {"log_level": "warn"}],
)
def test_invalid_values_are_config_errors(setting: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        load_settings(**setting)


def test_log_level_is_case_insensitive() -> None:
    assert load_settings(log_level="DEBUG").log_level == "debug"


def test_example_codex_config_points_at_the_responses_route() -> None:
    import tomllib

    example = Path(__file__).parent.parent / "examples" / "codex-config.toml"
    config = tomllib.loads(example.read_text())
    provider = config["model_providers"][config["model_provider"]]
    assert provider["wire_api"] == "responses"
    assert provider["base_url"].endswith("/v1")
    assert provider["name"] != "OpenAI"
