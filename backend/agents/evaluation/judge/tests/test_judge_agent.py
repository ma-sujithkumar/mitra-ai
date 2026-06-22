"""Tests for judge_agent.py: JSON extraction from LLM ranking responses.

All tests run without network or LLM access.
"""

from backend.agents.evaluation.judge.judge_agent import _extract_json_payload


class TestExtractJsonPayload:

    def test_pure_json_passes_through(self) -> None:
        raw = '{"ranking": [], "overall_commentary": "x"}'
        assert _extract_json_payload(raw) == raw

    def test_strips_leading_and_trailing_whitespace(self) -> None:
        raw = '\n\n  {"ranking": [], "overall_commentary": "x"}  \n'
        assert _extract_json_payload(raw) == '{"ranking": [], "overall_commentary": "x"}'

    def test_extracts_fenced_json_block(self) -> None:
        raw = 'Here is my ranking:\n```json\n{"ranking": [], "overall_commentary": "x"}\n```\nDone.'
        assert _extract_json_payload(raw) == '{"ranking": [], "overall_commentary": "x"}'

    def test_extracts_unfenced_json_after_prose_preamble(self) -> None:
        # The exact failure mode observed live: the model writes an analysis
        # paragraph before the JSON object, with no markdown fence at all.
        raw = (
            "Now I'll analyze the results and provide the ranking.\n\n"
            "**Key Observations:**\n- ModelA leads on accuracy.\n\n"
            '{"ranking": [{"model_name": "ModelA", "rank": 1, "reasoning": "x", '
            '"shap_domain_correlation": "y", "flags": []}], "overall_commentary": "z"}'
        )
        extracted = _extract_json_payload(raw)
        assert extracted.startswith("{")
        assert extracted.endswith("}")
        assert '"model_name": "ModelA"' in extracted

    def test_no_json_present_returns_stripped_text(self) -> None:
        raw = "  no json here at all  "
        assert _extract_json_payload(raw) == "no json here at all"
