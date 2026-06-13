import logging
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from config_loader import ConfigLoader
from mini_data import MiniDataGenerator
from session import SessionManager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target_col: Optional[str] = Form(default=""),
    problem_type: Optional[str] = Form(default="auto"),
    description: Optional[str] = Form(default=""),
) -> JSONResponse:
    allowed_extensions = ConfigLoader.get_str("upload", "ALLOWED_EXTENSIONS").split(",")
    max_size_mb = ConfigLoader.get_int("upload", "MAX_FILE_SIZE_MB")

    file_suffix = Path(file.filename).suffix.lower()
    if file_suffix not in [ext.strip() for ext in allowed_extensions]:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{file_suffix}' not allowed. Accepted: {', '.join(allowed_extensions)}",
        )

    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File size {file_size_mb:.1f} MB exceeds limit of {max_size_mb} MB",
        )

    session_id = SessionManager.create_session()
    data_path = SessionManager.get_session_path(session_id, "data/data.csv")

    with open(data_path, "wb") as dest_file:
        dest_file.write(content)

    logger.info(f"=> Uploaded {file.filename} ({file_size_mb:.2f} MB) to session {session_id}")

    generator = MiniDataGenerator()
    generator.generate(session_id, data_path)

    return JSONResponse(content={"session_id": session_id, "filename": file.filename})
