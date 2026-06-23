from backend.agents.llm_smoke_test import LlmSmokeTester
from backend.llm_failures import BILLING_INSUFFICIENT_HINT
from backend.llm_failures import has_llm_billing_error
from backend.llm_failures import has_llm_quota_error


# Mirrors the provider payload Anthropic returns through litellm when the
# account has run out of credit (surfaced as a plain BadRequestError).
ANTHROPIC_CREDIT_BALANCE_MESSAGE = (
    "litellm.BadRequestError: AnthropicException - "
    '{"type":"error","error":{"type":"invalid_request_error","message":'
    '"Your credit balance is too low to access the Anthropic API. Please go '
    'to Plans & Billing to upgrade or purchase credits."}}'
)


def test_billing_error_detected_for_credit_balance_message() -> None:
    exception = ValueError(ANTHROPIC_CREDIT_BALANCE_MESSAGE)

    assert has_llm_billing_error(exception=exception) is True
    # A credit-balance failure is billing, not a rate-limit/quota throttle.
    assert has_llm_quota_error(exception=exception) is False


def test_smoke_test_failure_message_reports_billing() -> None:
    exception = ValueError(ANTHROPIC_CREDIT_BALANCE_MESSAGE)

    message = LlmSmokeTester._failure_message(exception=exception)

    assert message == BILLING_INSUFFICIENT_HINT


def test_billing_detector_ignores_unrelated_errors() -> None:
    exception = ValueError("connection reset by peer")

    assert has_llm_billing_error(exception=exception) is False
