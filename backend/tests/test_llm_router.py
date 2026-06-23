from dotenv import dotenv_values
from fastapi.testclient import TestClient

from backend.agents.llm_smoke_test import LlmSmokeTestError
from backend.agents.llm_smoke_test import LlmSmokeTestResult
from backend.agents.metadata_gen_agent import LlmSettings
from backend.config_loader import ConfigLoader
from backend.main import create_app


class PassingSmokeTester:
    def __init__(self) -> None:
        self.requests: list[LlmSettings] = []

    def run(self, llm_settings: LlmSettings) -> LlmSmokeTestResult:
        self.requests.append(llm_settings)
        return LlmSmokeTestResult(
            provider=llm_settings.provider,
            model=llm_settings.model,
            latency_ms=17,
        )


class FailingSmokeTester:
    def run(self, llm_settings: LlmSettings) -> LlmSmokeTestResult:
        raise LlmSmokeTestError("provider rejected credentials")


def test_llm_smoke_test_persists_verified_settings_to_env(
    test_config_loader: ConfigLoader,
) -> None:
    env_path = test_config_loader.repo_root / ".env"
    env_path.write_text(
        "AUTHDB_HOST=db.internal\n"
        "LLM_TYPE=openai\n"
        "LLM_API_KEY=old-key\n"
        "LLM_CA_BUNDLE=ca.pem\n",
        encoding="utf-8",
    )
    app = create_app(config_loader=test_config_loader)
    fake_smoke_tester = PassingSmokeTester()
    app.state.llm_smoke_tester = fake_smoke_tester
    client = TestClient(app)

    response = client.post(
        "/api/llm/smoke-test",
        json={
            "provider": "anthropic",
            "model": "anthropic/run-model",
            "api_key": "new-secret-key",
            "gateway_url": "https://gateway.example.test",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "new-secret-key" not in response.text
    env_values = dotenv_values(env_path)
    assert env_values["AUTHDB_HOST"] == "db.internal"
    assert env_values["LLM_CA_BUNDLE"] == "ca.pem"
    assert env_values["LLM_TYPE"] == "anthropic"
    assert env_values["LLM_MODEL"] == "anthropic/run-model"
    assert env_values["LLM_API_KEY"] == "new-secret-key"
    assert env_values["LLM_GATEWAY_URL"] == "https://gateway.example.test"
    assert fake_smoke_tester.requests[0].source == "per_run"


def test_llm_smoke_test_failure_does_not_persist_settings(
    test_config_loader: ConfigLoader,
) -> None:
    env_path = test_config_loader.repo_root / ".env"
    original_env = (
        "LLM_TYPE=openai\n"
        "LLM_MODEL=openai/old-model\n"
        "LLM_API_KEY=old-key\n"
        "LLM_GATEWAY_URL=https://old.example.test\n"
    )
    env_path.write_text(original_env, encoding="utf-8")
    app = create_app(config_loader=test_config_loader)
    app.state.llm_smoke_tester = FailingSmokeTester()
    client = TestClient(app)

    response = client.post(
        "/api/llm/smoke-test",
        json={
            "provider": "anthropic",
            "model": "anthropic/new-model",
            "api_key": "new-secret-key",
            "gateway_url": "https://new.example.test",
        },
    )

    assert response.status_code == 502
    assert "new-secret-key" not in response.text
    assert env_path.read_text(encoding="utf-8") == original_env
