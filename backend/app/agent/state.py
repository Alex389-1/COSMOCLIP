from typing import List, Dict, Any, Optional, TypedDict
from backend.app.schemas.query import TraceStep, EvidenceRegion, ConfidenceInfo, ModelMetadata

class AgentState(TypedDict, total=False):
    # Inputs
    run_id: str
    question: str
    location_name: Optional[str]
    scene_id: Optional[str]
    image_data_url: Optional[str]
    bbox: Optional[List[float]]
    enable_grounding: bool
    enable_voice_response: bool
    session_id: str

    # Interpretation & Geocoding
    is_valid_input: bool
    validation_error: Optional[str]
    task_type: str                  # 'vqa', 'change_analysis', 'captioning', 'grounding'
    target_entity: Optional[str]
    is_comparison: bool
    location_meta: Optional[Dict[str, Any]]
    query_intent_meta: Dict[str, Any]

    # Preprocessed assets
    image_bytes: Optional[bytes]
    image_url: Optional[str]
    optical_url: Optional[str]
    sar_url: Optional[str]
    sar_vv_url: Optional[str]
    sar_vh_url: Optional[str]
    fused_url: Optional[str]
    sar_diff_url: Optional[str]
    sar_metrics: Optional[Dict[str, Any]]
    optical_metrics: Optional[Dict[str, Any]]
    before_image_url: Optional[str]
    after_image_url: Optional[str]
    heatmap_url: Optional[str]
    sar_heatmap_url: Optional[str]
    heatmap_bounds: Optional[List[List[float]]]
    cva_metrics: Optional[Dict[str, Any]]
    spectral_indices: Optional[Dict[str, float]]
    scene_metadata: Optional[Dict[str, Any]]

    # Model Execution Outputs
    web_intelligence: Optional[Dict[str, Any]]
    ground_truth_context: Optional[Dict[str, Any]]
    raw_answer: Optional[str]
    spoken_text: Optional[str]
    change_summary: Optional[Dict[str, Any]]
    confidence_info: Optional[ConfidenceInfo]
    evidence_regions: List[EvidenceRegion]
    model_metadata: Optional[ModelMetadata]

    # Trace & Final Output
    execution_trace: List[TraceStep]
    final_answer: Optional[str]
    audio_base64: Optional[str]
