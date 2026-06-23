"""Judge agent package (ranks candidate models, applies rule engine + LLM).

Import submodules explicitly to avoid eager heavy imports (google.adk) at
package-discovery time:

    from backend.agents.evaluation.judge.judge_agent import JudgeAgent
    from backend.agents.evaluation.judge.adapter import UpstreamAdapter
"""
