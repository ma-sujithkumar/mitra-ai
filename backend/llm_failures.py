from __future__ import annotations

import ssl


# Shown when LiteLLM cannot infer a provider from the model string. The model
# must carry a recognized prefix (e.g. an OpenAI-compatible gateway like NVIDIA
# needs 'openai/<model>').
PROVIDER_PREFIX_HINT = (
    "Model name is missing a recognized provider prefix. For OpenAI-compatible "
    "gateways such as NVIDIA (https://integrate.api.nvidia.com/v1), prefix the "
    "model with 'openai/', for example 'openai/google/gemma-2-2b-it'. Other "
    "prefixes: 'gemini/', 'anthropic/', 'ollama/', 'huggingface/'."
)


# Shown when the model rejects the function-calling 'tools' parameter that the
# metadata agent requires (small instruct models often lack tool support).
TOOL_CALLING_UNSUPPORTED_HINT = (
    "The selected model does not support tool/function calling, which metadata "
    "generation requires. Choose a tool-calling-capable model on your provider, "
    "for example 'openai/meta/llama-3.3-70b-instruct' or "
    "'openai/nvidia/llama-3.1-nemotron-70b-instruct' on NVIDIA."
)


# Shown when the provider rejects the request because the account has no usable
# credit balance. Providers return this as a plain 400 (e.g. Anthropic's
# "credit balance is too low") rather than a rate-limit/quota error, so it needs
# its own detector to avoid the generic catch-all message.
BILLING_INSUFFICIENT_HINT = (
    "LLM account has insufficient credit/billing balance. Add credits or billing "
    "to the provider account for this API key, or use a funded key, provider, or "
    "gateway."
)


def iter_exception_chain(exception: BaseException) -> list[BaseException]:
    visited_exception_ids: set[int] = set()
    current_exception: BaseException | None = exception
    exception_chain: list[BaseException] = []
    while current_exception is not None:
        current_exception_id = id(current_exception)
        if current_exception_id in visited_exception_ids:
            break
        visited_exception_ids.add(current_exception_id)
        exception_chain.append(current_exception)

        current_exception = current_exception.__cause__ or current_exception.__context__
    return exception_chain


def has_llm_quota_error(exception: BaseException) -> bool:
    for current_exception in iter_exception_chain(exception=exception):
        exception_class_name = current_exception.__class__.__name__.lower()
        exception_message = str(current_exception).lower()
        if "ratelimit" in exception_class_name or "rate_limit" in exception_message:
            return True
        if "insufficient_quota" in exception_message:
            return True
        if "exceeded your current quota" in exception_message:
            return True
    return False


def has_llm_billing_error(exception: BaseException) -> bool:
    # Detects provider responses that signal an empty/insufficient billing
    # balance (distinct from rate limits and per-minute quota throttling).
    billing_phrases = [
        "credit balance is too low",
        "credit balance",
        "purchase credits",
        "plans & billing",
        "billing",
        "payment required",
    ]
    for current_exception in iter_exception_chain(exception=exception):
        exception_message = str(current_exception).lower()
        if any(phrase in exception_message for phrase in billing_phrases):
            return True
    return False


def has_ssl_certificate_error(exception: BaseException) -> bool:
    for current_exception in iter_exception_chain(exception=exception):
        if isinstance(current_exception, ssl.SSLCertVerificationError):
            return True
        if "SSLCertVerificationError" in current_exception.__class__.__name__:
            return True
    return False


def has_llm_provider_missing_error(exception: BaseException) -> bool:
    for current_exception in iter_exception_chain(exception=exception):
        exception_message = str(current_exception).lower()
        if "llm provider not provided" in exception_message:
            return True
        if "llm provider not found" in exception_message:
            return True
    return False


def has_tool_calling_unsupported_error(exception: BaseException) -> bool:
    for current_exception in iter_exception_chain(exception=exception):
        exception_message = str(current_exception).lower()
        mentions_tools = "tools" in exception_message or "function calling" in exception_message
        if not mentions_tools:
            continue
        if "extra_forbidden" in exception_message:
            return True
        if "extra inputs are not permitted" in exception_message:
            return True
        if "not permitted" in exception_message:
            return True
        if "not support" in exception_message or "unsupported" in exception_message:
            return True
    return False


def has_llm_authentication_error(exception: BaseException) -> bool:
    for current_exception in iter_exception_chain(exception=exception):
        exception_class_name = current_exception.__class__.__name__.lower()
        exception_message = str(current_exception).lower()
        if "authentication" in exception_class_name:
            return True
        if "invalid api key" in exception_message or "invalid_api_key" in exception_message:
            return True
        if "incorrect api key" in exception_message:
            return True
        if "no api key" in exception_message or "missing api key" in exception_message:
            return True
    return False
