# Plan: Fix Metadata Generation Failure

This plan addresses the "Metadata generation failed" error during the initial pipeline run.

## 1. Problem Diagnosis
- When the Gemini model invokes the tool `write_metadata(session_id, metadata)`, it passes the bookkeeping `session_id` as the top-level argument, and does not duplicate it inside the `metadata` dictionary.
- The `MetadataAgentToolAdapter.write_metadata` receives `session_id` and the `metadata` dictionary, but does not inject `session_id` into the `metadata` dictionary.
- `MetadataTools.write_metadata` validates the `metadata` dictionary against `metadata_schema.json`, which lists `"session_id"` under `required`.
- Because `"session_id"` is missing from the dictionary, validation raises `jsonschema.exceptions.ValidationError: 'session_id' is a required property`.
- The ADK agent retry loop catches this and retries up to 3 times (configured via `LLM_MAX_RETRIES=3` in `config.ini`). After 3 attempts, it terminates without having successfully written `metadata.json`.
- The FastAPI metadata router catches this failure but does not log the traceback, swallowing the exception and displaying a generic `"Metadata generation failed"` error.

## 2. Proposed Changes
### A. Inject `session_id` in `MetadataAgentToolAdapter.write_metadata`
- In [backend/agents/metadata_gen_agent.py](file:///home/sujithma/mitra/backend/agents/metadata_gen_agent.py), inside `MetadataAgentToolAdapter.write_metadata`, ensure we always inject the `session_id` into `normalized_metadata`.
```python
    def write_metadata(
        self,
        session_id: str,
        metadata: dict[str, Any],
    ) -> dict[str, str]:
        normalized_metadata = self._coerce_metadata_dict(metadata=metadata)
        # Ensure session_id is always present in normalized_metadata
        normalized_metadata["session_id"] = session_id
        ...
```

### B. Add Traceback Logging in Metadata Router
- In [backend/routers/metadata.py](file:///home/sujithma/mitra/backend/routers/metadata.py), import `logging`, set up a router-specific logger, and log the exception with traceback in `_metadata_generation_failed`.
```python
import logging

logger = logging.getLogger("mitra.metadata_router")
...
def _metadata_generation_failed(
    metadata_request: MetadataRequest,
    job_registry: JobRegistry,
    exception: Exception,
) -> None:
    logger.exception(
        "Metadata generation failed for session: %s",
        metadata_request.session_id,
    )
    failure_message = _metadata_failure_message(exception=exception)
    ...
```

## 3. Verification Plan
- Run our reproduction script `claude_scripts/reproduce_metadata.py` and verify it now succeeds.
- Run all backend unit tests: `~/venv/bin/pytest backend/` to ensure no regressions.
- Verify the metadata generation works end-to-end.
