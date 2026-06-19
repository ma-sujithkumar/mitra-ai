import json
import ssl
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from backend.agents.metadata_gen_agent import MetadataGenerationError
from backend.agents.metadata_gen_agent import MetadataGenerationInput
from backend.agents.metadata_gen_agent import MetadataGenerationResult
from backend.agents.tools import MetadataTools
from backend.config_loader import ConfigLoader
from backend.main import create_app


class SuccessfulMetadataRunner:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.requests: list[MetadataGenerationInput] = []

    def generate_metadata(
        self,
        generation_input: MetadataGenerationInput,
    ) -> MetadataGenerationResult:
        self.requests.append(generation_input)
        metadata = valid_metadata(session_id=generation_input.session_id)
        result = MetadataTools(workspace_root=self.workspace_root).write_metadata(
            session_id=generation_input.session_id,
            metadata=metadata,
        )
        return MetadataGenerationResult(
            metadata=metadata,
            metadata_path=result.metadata_path,
        )


class FailingMetadataRunner:
    def generate_metadata(
        self,
        generation_input: MetadataGenerationInput,
    ) -> MetadataGenerationResult:
        raise MetadataGenerationError("invalid credentials")


class CertificateFailingMetadataRunner:
    def generate_metadata(
        self,
        generation_input: MetadataGenerationInput,
    ) -> MetadataGenerationResult:
        certificate_error = ssl.SSLCertVerificationError(
            "unable to get local issuer certificate"
        )
        raise MetadataGenerationError("connection error") from certificate_error


class RateLimitError(Exception):
    pass


class QuotaFailingMetadataRunner:
    def generate_metadata(
        self,
        generation_input: MetadataGenerationInput,
    ) -> MetadataGenerationResult:
        provider_error = RateLimitError(
            "RateLimitError: OpenAIException - insufficient_quota"
        )
        raise MetadataGenerationError("provider request failed") from provider_error


def valid_metadata(session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "problem_type": "supervised",
        "problem_subtype": "classification",
        "target_col": "target",
        "target_col_type": "categorical",
        "input_cols": [
            {
                "name": "feature_one",
                "col_type": "numeric",
            }
        ],
        "cols_to_drop": [],
        "statistics": {
            "feature_one": {
                "count": 10,
                "mean": 5.5,
                "std": 3.02,
                "min": 1,
                "25%": 3.25,
                "50%": 5.5,
                "75%": 7.75,
                "max": 10,
                "top": None,
                "freq": None,
            }
        },
    }


def upload_and_validate(client: TestClient) -> str:
    upload_response = client.post(
        "/api/upload",
        files={
            "dataset_file": (
                "iris.csv",
                (
                    b"feature_one,feature_two,target\n"
                    b"1,2,a\n2,3,a\n3,4,b\n4,5,b\n5,6,c\n6,7,c\n"
                    b"7,8,a\n8,9,b\n9,10,c\n10,11,a\n"
                ),
                "text/csv",
            )
        },
    )
    assert upload_response.status_code == 200
    session_id = upload_response.json()["session_id"]
    validation_response = client.post(
        "/api/validate",
        json={
            "session_id": session_id,
            "target_col": "target",
        },
    )
    assert validation_response.status_code == 200
    return session_id


def test_metadata_cannot_start_before_validation(
    test_config_loader: ConfigLoader,
) -> None:
    client = TestClient(create_app(config_loader=test_config_loader))
    upload_response = client.post(
        "/api/upload",
        files={
            "dataset_file": (
                "iris.csv",
                b"feature_one,target\n1,a\n2,b\n",
                "text/csv",
            )
        },
    )
    session_id = upload_response.json()["session_id"]

    response = client.post(
        "/api/metadata",
        json={
            "session_id": session_id,
            "provider": "openai",
            "api_key": "secret-key",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "VALIDATION_REQUIRED"
    assert "secret-key" not in response.text


def test_metadata_starts_after_passing_validation(
    test_config_loader: ConfigLoader,
) -> None:
    app = create_app(config_loader=test_config_loader)
    fake_runner = SuccessfulMetadataRunner(
        workspace_root=test_config_loader.paths.workspace_root
    )
    app.state.metadata_agent_runner = fake_runner
    client = TestClient(app)
    session_id = upload_and_validate(client=client)

    response = client.post(
        "/api/metadata",
        json={
            "session_id": session_id,
            "description": "Predict the target class.",
            "target_col": "target",
            "problem_type": "classification",
            "provider": "openai",
            "model": "",
            "api_key": "secret-key",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "accepted"
    assert response.json()["session_id"] == session_id
    assert "secret-key" not in response.text
    assert fake_runner.requests[0].description == "Predict the target class."
    reports_dir = test_config_loader.paths.workspace_root / session_id / "reports"
    metadata = json.loads((reports_dir / "metadata.json").read_text("utf-8"))
    assert metadata["session_id"] == session_id


def test_metadata_missing_credentials_returns_503(
    test_config_loader: ConfigLoader,
) -> None:
    client = TestClient(create_app(config_loader=test_config_loader))
    session_id = upload_and_validate(client=client)

    response = client.post(
        "/api/metadata",
        json={
            "session_id": session_id,
            "provider": "openai",
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "LLM_CREDENTIALS_REQUIRED"


def test_metadata_invalid_ca_bundle_returns_configuration_error(
    test_config_loader: ConfigLoader,
) -> None:
    env_path = test_config_loader.repo_root / ".env"
    env_path.write_text(
        "LLM_CA_BUNDLE=invalid-ca.pem\n",
        encoding="utf-8",
    )
    (test_config_loader.repo_root / "invalid-ca.pem").write_text(
        "not a certificate",
        encoding="utf-8",
    )
    client = TestClient(create_app(config_loader=test_config_loader))
    session_id = upload_and_validate(client=client)

    response = client.post(
        "/api/metadata",
        json={
            "session_id": session_id,
            "provider": "openai",
            "api_key": "secret-key",
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "LLM_CONFIGURATION_UNAVAILABLE"
    assert "LLM_CA_BUNDLE must point to a PEM file" in response.json()["detail"][
        "message"
    ]
    assert "secret-key" not in response.text


def test_metadata_events_stream_progress_and_done(
    test_config_loader: ConfigLoader,
) -> None:
    app = create_app(config_loader=test_config_loader)
    app.state.metadata_agent_runner = SuccessfulMetadataRunner(
        workspace_root=test_config_loader.paths.workspace_root
    )
    client = TestClient(app)
    session_id = upload_and_validate(client=client)

    client.post(
        "/api/metadata",
        json={
            "session_id": session_id,
            "provider": "openai",
            "api_key": "secret-key",
        },
    )
    response = client.get(f"/api/metadata/events?session_id={session_id}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"type": "progress"' in response.text
    assert '"type": "done"' in response.text
    assert "secret-key" not in response.text


def test_metadata_runner_failure_returns_503_without_secret(
    test_config_loader: ConfigLoader,
) -> None:
    app = create_app(config_loader=test_config_loader)
    app.state.metadata_agent_runner = FailingMetadataRunner()
    client = TestClient(app)
    session_id = upload_and_validate(client=client)

    response = client.post(
        "/api/metadata",
        json={
            "session_id": session_id,
            "provider": "openai",
            "api_key": "secret-key",
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "METADATA_GENERATION_FAILED"
    assert "secret-key" not in response.text
    events_response = client.get(f"/api/metadata/events?session_id={session_id}")
    assert '"type": "error"' in events_response.text
    assert "secret-key" not in events_response.text


def test_metadata_certificate_failure_returns_actionable_message(
    test_config_loader: ConfigLoader,
) -> None:
    app = create_app(config_loader=test_config_loader)
    app.state.metadata_agent_runner = CertificateFailingMetadataRunner()
    client = TestClient(app)
    session_id = upload_and_validate(client=client)

    response = client.post(
        "/api/metadata",
        json={
            "session_id": session_id,
            "provider": "openai",
            "api_key": "secret-key",
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "METADATA_GENERATION_FAILED"
    assert "LLM HTTPS certificate verification failed" in response.json()["detail"][
        "message"
    ]
    assert "secret-key" not in response.text


def test_metadata_quota_failure_returns_actionable_message(
    test_config_loader: ConfigLoader,
) -> None:
    app = create_app(config_loader=test_config_loader)
    app.state.metadata_agent_runner = QuotaFailingMetadataRunner()
    client = TestClient(app)
    session_id = upload_and_validate(client=client)

    response = client.post(
        "/api/metadata",
        json={
            "session_id": session_id,
            "provider": "openai",
            "api_key": "secret-key",
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "METADATA_GENERATION_FAILED"
    assert "LLM provider quota exceeded or rate limited" in response.json()["detail"][
        "message"
    ]
    assert "secret-key" not in response.text
