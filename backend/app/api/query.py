from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from backend.app.schemas.query import QueryRequest, QueryResponse
from backend.app.agent.controller import SatQueryAgentController
from backend.app.agent.registry import GLOBAL_REGISTRY

router = APIRouter(prefix="/api", tags=["Query & Reasoning"])
controller = SatQueryAgentController()

# In-memory execution trace store
RUNS_STORE: Dict[str, QueryResponse] = {}

class CurrentViewQuery(BaseModel):
    question: str = Field(..., description="Question about the currently visible map view")
    image_base64: str = Field(..., description="Direct canvas screenshot of active map view (JPEG base64)")
    session_id: Optional[str] = Field("default_session", description="Session identifier")

@router.post("/query/current-view", response_model=QueryResponse)
async def query_current_view(payload: CurrentViewQuery):
    """
    Direct endpoint for analyzing a client-side screenshot of the active map canvas.
    Completely bypasses geocoding, STAC imagery retrieval, and map recentering.
    """
    try:
        response = await controller.run_current_view(
            question=payload.question,
            image_base64=payload.image_base64,
            session_id=payload.session_id or "default_session"
        )
        RUNS_STORE[response.run_id] = response
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Current-view analysis failed: {str(e)}")

@router.post("/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest):
    """
    Primary endpoint for natural-language question answering over satellite imagery.
    Dispatches between navigation (geocoding + recenter) and current-view (screenshot VLM).
    """
    try:
        # If client provided an active map screenshot and query is not an explicit navigation destination
        if request.image_base64:
            from backend.app.agent.router import classify
            if classify(request.question) != "navigation":
                response = await controller.run_current_view(
                    question=request.question,
                    image_base64=request.image_base64,
                    session_id=request.session_id or "default_session"
                )
                RUNS_STORE[response.run_id] = response
                return response

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
