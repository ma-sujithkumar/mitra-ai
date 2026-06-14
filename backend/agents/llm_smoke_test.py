from __future__ import annotations

import time
from dataclasses import dataclass

import litellm

from backend.agents.metadata_gen_agent import LlmSettings
from backend.agents.metadata_gen_agent import configure_default_ssl_certificates
from backend.llm_failures import PROVIDER_PREFIX_HINT
from backend.llm_failures import TOOL_CALLING_UNSUPPORTED_HINT
from backend.llm_failures import has_llm_authentication_error
from backend.llm_failures import has_llm_provider_missing_error
from backend.llm_failures import has_llm_quota_error
from backend.llm_failures import has_ssl_certificate_error
from backend.llm_failures import has_tool_calling_unsupported_error


# Minimal prompt used to confirm the provider/model/credentials respond.
SMOKE_TEST_PROMPT = "ping"
SMOKE_TEST_MAX_TOKENS = 16

# Probe tool sent on the smoke-test request so it validates the same
# function-calling capability the metadata agent depends on.
SMOKE_TEST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "connectivity_probe",
            "description": "Connectivity probe used to verify tool-calling support.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


@dataclass(frozen=True)
class LlmSmokeTestResult:
    provider: str
    model: str
    latency_ms: int


class LlmSmokeTestError(RuntimeError):
    pass


class LlmSmokeTester:
    """Issues a single minimal completion to verify LLM connectivity."""

    def run(self, llm_settings: LlmSettings) -> LlmSmokeTestResult:
        configure_default_ssl_certificates(ca_bundle_path=llm_settings.ca_bundle_path)
        completion_kwargs = self._build_completion_kwargs(llm_settings=llm_settings)

        started_at_epoch = time.monotonic()
        try:
            litellm.completion(
                model=llm_settings.model,
                messages=[{"role": "user", "content": SMOKE_TEST_PROMPT}],
                max_tokens=SMOKE_TEST_MAX_TOKENS,
                tools=SMOKE_TEST_TOOLS,
                **completion_kwargs,
            )
        except Exception as exception:
            raise LlmSmokeTestError(
                self._failure_message(exception=exception)
            ) from exception

        latency_ms = int((time.monotonic() - started_at_epoch) * 1000)
        return LlmSmokeTestResult(
            provider=llm_settings.provider,
            model=llm_settings.model,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _build_completion_kwargs(llm_settings: LlmSettings) -> dict[str, str]:
        completion_kwargs: dict[str, str] = {}
        if llm_settings.api_key:
            completion_kwargs["api_key"] = llm_settings.api_key
        if llm_settings.gateway_url:
            completion_kwargs["api_base"] = llm_settings.gateway_url
        return completion_kwargs

    @staticmethod
    def _failure_message(exception: Exception) -> str:
        if has_llm_provider_missing_error(exception=exception):
            return PROVIDER_PREFIX_HINT
        if has_tool_calling_unsupported_error(exception=exception):
            return TOOL_CALLING_UNSUPPORTED_HINT
        if has_llm_authentication_error(exception=exception):
            return (
                "LLM authentication failed. Check the API key for the selected "
                "provider, model, and gateway."
            )
        if has_llm_quota_error(exception=exception):
            return (
                "LLM provider quota exceeded or rate limited. Check the provider "
                "billing/quota for this API key, or choose another key, model, "
                "provider, or gateway."
            )
        if has_ssl_certificate_error(exception=exception):
            return (
                "LLM HTTPS certificate verification failed. Configure LLM_CA_BUNDLE "
                "with a PEM bundle containing your local root CA and restart the backend."
            )
        return "LLM connection test failed. Verify the provider, model, and credentials."
