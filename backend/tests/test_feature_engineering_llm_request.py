from types import SimpleNamespace

from google.adk.models.lite_llm import _get_completion_inputs
from google.genai import types as genai_types

from backend.agents.feature_engineering import orchestrator
from backend.agents.metadata_gen_agent import LlmSettings


class CapturingLlm:
    def __init__(self) -> None:
        self.generation_params: dict[str, object] | None = None

    async def generate_content_async(self, request, stream=False):
        del stream
        _, _, _, generation_params = await _get_completion_inputs(request, request.model)
        self.generation_params = generation_params
        yield SimpleNamespace(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(text="ok")],
            )
        )


def test_feature_engineering_model_call_sets_generation_params_for_adk(
    monkeypatch,
) -> None:
    fake_llm = CapturingLlm()
    monkeypatch.setattr(
        orchestrator,
        "build_llm_model",
        lambda settings: fake_llm,
    )
    llm_settings = LlmSettings(
        provider="openai",
        model="openai/test-model",
        api_key="test-key",
    )

    model_call = orchestrator._make_model_call(
        llm_settings=llm_settings,
        max_tokens=16,
    )

    assert model_call("Respond with ok") == "ok"
    assert fake_llm.generation_params == {
        "temperature": 0.0,
        "max_completion_tokens": 16,
    }
