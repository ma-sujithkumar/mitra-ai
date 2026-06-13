from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from backend.config_loader import ConfigLoader
from backend.main import create_app


def test_upload_csv_creates_session_and_artifacts(
    test_config_loader: ConfigLoader,
) -> None:
    client = TestClient(create_app(config_loader=test_config_loader))

    response = client.post(
        "/api/upload",
        files={
            "dataset_file": (
                "iris.csv",
                b"a,b,target\n1,2,x\n3,4,y\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    session_path = test_config_loader.paths.workspace_root / payload["session_id"]
    assert (session_path / "data" / "data.csv").is_file()
    assert (session_path / "data" / "mini_data.csv").is_file()
    assert payload["summary"]["row_count"] == 2
    assert payload["summary"]["column_count"] == 3


def test_upload_excel_preserves_source_and_writes_canonical_csv(
    test_config_loader: ConfigLoader,
) -> None:
    client = TestClient(create_app(config_loader=test_config_loader))
    excel_buffer = BytesIO()
    pd.DataFrame({"a": [1, 2], "target": ["x", "y"]}).to_excel(
        excel_buffer,
        index=False,
    )

    response = client.post(
        "/api/upload",
        files={
            "dataset_file": (
                "data.xlsx",
                excel_buffer.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    session_path = test_config_loader.paths.workspace_root / payload["session_id"]
    assert (session_path / "data" / "source.xlsx").is_file()
    assert (session_path / "data" / "data.csv").is_file()


def test_upload_optional_metadata_file(test_config_loader: ConfigLoader) -> None:
    client = TestClient(create_app(config_loader=test_config_loader))

    response = client.post(
        "/api/upload",
        files={
            "dataset_file": ("data.csv", b"a,target\n1,x\n2,y\n", "text/csv"),
            "metadata_file": (
                "metadata.json",
                b"{\"target\":\"target\"}",
                "application/json",
            ),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    session_path = test_config_loader.paths.workspace_root / payload["session_id"]
    assert (session_path / "data" / "user_metadata.json").is_file()


def test_upload_rejects_unsupported_extension(test_config_loader: ConfigLoader) -> None:
    client = TestClient(create_app(config_loader=test_config_loader))

    response = client.post(
        "/api/upload",
        files={
            "dataset_file": (
                "images.zip",
                b"not supported",
                "application/zip",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "UNSUPPORTED_FILE_EXTENSION"


def test_recent_uploads_returns_latest_five_only(
    test_config_loader: ConfigLoader,
) -> None:
    client = TestClient(create_app(config_loader=test_config_loader))
    for upload_index in range(6):
        client.post(
            "/api/upload",
            files={
                "dataset_file": (
                    f"data_{upload_index}.csv",
                    f"a,target\n{upload_index},x\n{upload_index + 1},y\n".encode(
                        "utf-8"
                    ),
                    "text/csv",
                )
            },
        )

    response = client.get("/api/uploads/recent?limit=5")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["uploads"]) == 5
    assert payload["uploads"][0]["original_filename"] == "data_5.csv"
