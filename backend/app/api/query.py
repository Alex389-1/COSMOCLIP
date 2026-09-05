from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from backend.app.schemas.query import QueryRequest, QueryResponse
from backend.app.agent.controller import SatQueryAgentController
from backend.app.agent.registry import GLOBAL_REGISTRY

router = APIRouter(prefix="/api", tags=["Query & Reasoning"])
controller = SatQueryAgentController()

# In-memory execution trace store
RUNS_STORE: Dict[str, QueryResponse] = {}

@router.post("/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest):
    """
    Primary endpoint for natural-language question answering over satellite imagery.
    Executes the LangGraph agentic controller:
    validate -> interpret -> preprocess -> RS-LLaVA VQA -> calibrate -> response + trace.
    """
    try:
        response = await controller.run(request.model_dump())
        RUNS_STORE[response.run_id] = response
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Controller execution failure: {str(e)}")

@router.get("/runs/{run_id}", response_model=QueryResponse)
async def get_execution_run(run_id: str):
    """
    Retrieve full execution trace and metadata for an audit or verification run.
    """
    if run_id not in RUNS_STORE:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return RUNS_STORE[run_id]

@router.get("/models/registry")
async def get_model_registry():
    """
    List all active specialist models and tool capability contracts.
    """
    tools = GLOBAL_REGISTRY.list_all_tools()
    return {
        "status": "ok",
        "count": len(tools),
        "tools": [t.model_dump() for t in tools]
    }
