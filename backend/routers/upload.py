from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import UploadFile

from backend.config_loader import ConfigLoader
from backend.dependencies import get_config_loader
from backend.dependencies import get_session_manager
from backend.mini_data import DatasetNormalizer
from backend.mini_data import UnsupportedDatasetTypeError
from backend.session import SessionManager


router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload")
def upload_dataset(
    dataset_file: UploadFile,
    metadata_file: UploadFile | None = None,
    config_loader: ConfigLoader = Depends(get_config_loader),
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, object]:
    dataset_extension = _validate_upload_file(
        upload_file=dataset_file,
        allowed_extensions=config_loader.upload.allowed_extensions,
        file_role="dataset",
    )
    session_info = session_manager.create_session(
        original_filename=dataset_file.filename or "dataset.csv"
    )
    source_upload_path = session_info.data_dir / f"upload{dataset_extension}"
    _save_upload_file(upload_file=dataset_file, destination_path=source_upload_path)
    _validate_file_size(
        file_path=source_upload_path,
        max_file_size_mb=config_loader.upload.max_file_size_mb,
    )

    if metadata_file is not None and metadata_file.filename:
        metadata_extension = _validate_upload_file(
            upload_file=metadata_file,
            allowed_extensions=[".csv", ".json"],
            file_role="metadata",
        )
        metadata_path = session_info.data_dir / f"user_metadata{metadata_extension}"
        _save_upload_file(upload_file=metadata_file, destination_path=metadata_path)

    normalizer = DatasetNormalizer(
        mini_data_sample_rows=config_loader.upload.mini_data_sample_rows,
        chunk_size_rows=config_loader.upload.chunk_size_rows,
    )
    try:
        summary = normalizer.normalize(
            source_file=source_upload_path,
            data_dir=session_info.data_dir,
        )
    except UnsupportedDatasetTypeError as error:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "UNSUPPORTED_FILE_EXTENSION",
                "message": str(error),
            },
        ) from error

    session_manager.write_session_metadata(
        session_id=session_info.session_id,
        updates={
            "canonical_data_path": str(summary.canonical_data_path),
            "mini_data_path": str(summary.mini_data_path),
            "row_count": summary.row_count,
            "column_count": summary.column_count,
            "columns": summary.columns,
            "file_size_bytes": summary.file_size_bytes,
            "data_type": summary.data_type,
        },
    )

    return {
        "session_id": session_info.session_id,
        "original_filename": session_info.original_filename,
        "uploaded_at": session_info.uploaded_at,
        "summary": {
            "row_count": summary.row_count,
            "column_count": summary.column_count,
            "columns": summary.columns,
            "file_size_bytes": summary.file_size_bytes,
            "data_type": summary.data_type,
        },
    }


@router.get("/uploads/recent")
def recent_uploads(
    limit: int | None = None,
    config_loader: ConfigLoader = Depends(get_config_loader),
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, object]:
    resolved_limit = limit or config_loader.upload.recent_upload_limit
    return {
        "uploads": session_manager.list_recent_uploads(limit=resolved_limit),
    }


def _validate_upload_file(
    upload_file: UploadFile,
    allowed_extensions: list[str],
    file_role: str,
) -> str:
    filename = upload_file.filename or ""
    file_extension = Path(filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "UNSUPPORTED_FILE_EXTENSION",
                "message": f"Unsupported {file_role} extension: {file_extension}",
            },
        )
    return file_extension


def _validate_file_size(file_path: Path, max_file_size_mb: int) -> None:
    max_file_size_bytes = max_file_size_mb * 1024 * 1024
    if file_path.stat().st_size > max_file_size_bytes:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "FILE_TOO_LARGE",
                "message": f"File exceeds {max_file_size_mb} MB",
            },
        )


def _save_upload_file(upload_file: UploadFile, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("wb") as output_file:
        shutil.copyfileobj(upload_file.file, output_file)
