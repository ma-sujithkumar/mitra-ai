import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import litellm
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import upload, validate, metadata, runs, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ENV_PATH = Path(__file__).parent.parent / ".env"

# Global flag set during lifespan startup
llm_smoke_test_passed: bool = False


async def _run_llm_smoke_test() -> bool:
    llm_type = os.environ.get("LLM_TYPE", "")
    llm_api_key = os.environ.get("LLM_API_KEY", "")
    llm_gateway_url = os.environ.get("LLM_GATEWAY_URL", "")

    if not llm_api_key:
        logger.warning("=> LLM smoke-test: SKIPPED (LLM_API_KEY not set in .env)")
        return False

    model_map = {
        "anthropic": "anthropic/claude-haiku-4-5-20251001",
        "openai":    "openai/gpt-4o-mini",
        "gemini":    "gemini/gemini-2.0-flash",
    }
    model = model_map.get(llm_type, "anthropic/claude-haiku-4-5-20251001")

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word OK."}],
        "max_tokens": 10,
    }
    if llm_api_key:
        kwargs["api_key"] = llm_api_key
    if llm_gateway_url:
        kwargs["api_base"] = llm_gateway_url

    try:
        response = await litellm.acompletion(**kwargs)
        response_text = response.choices[0].message.content or ""
        if "ok" in response_text.lower():
            logger.info("=> LLM smoke-test: OK")
            return True
        else:
            logger.warning(f"=> LLM smoke-test: UNEXPECTED RESPONSE: {response_text!r}")
            return False
    except Exception as exc:
        logger.error(f"=> LLM smoke-test: FAILED — {exc}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_smoke_test_passed
    load_dotenv(ENV_PATH)
    logger.info("=> MITRA backend starting up")
    llm_smoke_test_passed = await _run_llm_smoke_test()
    yield
    logger.info("=> MITRA backend shutting down")


app = FastAPI(title="MITRA AI Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api")
app.include_router(validate.router, prefix="/api")
app.include_router(metadata.router, prefix="/api")
app.include_router(runs.router, prefix="/api")
app.include_router(health.router, prefix="/api")
