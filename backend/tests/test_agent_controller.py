import pytest
from backend.app.agent.controller import SatQueryAgentController
from backend.app.agent.router import QueryInterpreter
from backend.app.agent.registry import GLOBAL_REGISTRY

def test_query_interpreter_intent():
    res1 = QueryInterpreter.interpret("Is there a large water body in this area?")
    assert res1["task"] == "vqa"
    assert res1["target"] in ["water", "water_body", "general"]

    res2 = QueryInterpreter.interpret("Describe this satellite scene and urban density")
    assert res2["task"] == "captioning" or res2["target"] in ["urban", "general"]

    res3 = QueryInterpreter.interpret("Where is the Bhadla solar park?")
    assert res3["target"] in ["solar_park", "infrastructure", "general"]

def test_tool_registry():
    tools = GLOBAL_REGISTRY.list_all_tools()
    assert len(tools) >= 3
    vqa_tool = GLOBAL_REGISTRY.get_tool("satellite_vqa")
    assert vqa_tool is not None
    assert "vqa" in vqa_tool.task_types

@pytest.mark.asyncio
async def test_agent_controller_execution_flow():
    controller = SatQueryAgentController()
    response = await controller.run({
        "question": "Is there a large water body in this area?",
        "scene_id": "S2A_MSIL2A_20260814T054641_N0500_R120_T43RER_20260814T084512",
        "enable_grounding": True
    })

    assert response.task == "vqa"
    assert response.confidence.score > 0.0
    assert len(response.trace) >= 4
    assert any("validate_input" in t.step for t in response.trace)
    assert any("satellite_vqa" in t.step for t in response.trace)
    assert len(response.answer) > 20
