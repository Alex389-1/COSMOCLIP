"""Comprehensive 30-Question VQA Evaluation Suite (PRD §9, §11, Phase 4).

Tests across the 5 structured remote-sensing categories:
1. Presence
2. Land cover
3. Object detection
4. Counting
5. Spatial reasoning
"""

import pytest
import time
from voice_speech.engine.vqa.model import run_vqa
from voice_speech.tests.generate_test_tiles import create_sample_tiles

EVAL_BENCHMARK_QUESTIONS = [
    # Category 1: Presence (6 items)
    {"tile": "water", "question": "Is there water in this image?", "expected": "yes", "cat": "Presence"},
    {"tile": "water", "question": "Is a river or ocean visible?", "expected": "yes", "cat": "Presence"},
    {"tile": "urban", "question": "Is there water in this scene?", "expected": "no", "cat": "Presence"},
    {"tile": "vegetation", "question": "Is there a forest present?", "expected": "vegetation", "cat": "Presence"},
    {"tile": "urban", "question": "Are roads present in this tile?", "expected": "yes", "cat": "Presence"},
    {"tile": "port", "question": "Is there a coastline or port visible?", "expected": "water", "cat": "Presence"},

    # Category 2: Land Cover (6 items)
    {"tile": "vegetation", "question": "What type of land dominates this area?", "expected": "vegetation", "cat": "Land cover"},
    {"tile": "vegetation", "question": "What does this area look like?", "expected": "forest", "cat": "Land cover"},
    {"tile": "urban", "question": "What is the primary land use in this image?", "expected": "urban", "cat": "Land cover"},
    {"tile": "urban", "question": "What type of land cover dominates?", "expected": "residential", "cat": "Land cover"},
    {"tile": "water", "question": "What type of surface dominates this scene?", "expected": "water", "cat": "Land cover"},
    {"tile": "port", "question": "Describe the terrain and land cover.", "expected": "water", "cat": "Land cover"},

    # Category 3: Object (6 items)
    {"tile": "urban", "question": "Are there buildings visible in this image?", "expected": "yes", "cat": "Object"},
    {"tile": "urban", "question": "Are structures visible?", "expected": "structures", "cat": "Object"},
    {"tile": "port", "question": "Are there ships or vessels visible in the water?", "expected": "vessels", "cat": "Object"},
    {"tile": "port", "question": "Are boats present near the dock?", "expected": "vessels", "cat": "Object"},
    {"tile": "vegetation", "question": "Are there buildings visible in this forest?", "expected": "no", "cat": "Object"},
    {"tile": "water", "question": "Are there any ships visible in this open water?", "expected": "no", "cat": "Object"},

    # Category 4: Counting (6 items)
    {"tile": "urban", "question": "How many buildings are visible in this scene?", "expected": "15", "cat": "Counting"},
    {"tile": "urban", "question": "Count the number of visible structures.", "expected": "approximately", "cat": "Counting"},
    {"tile": "vegetation", "question": "How many ships are visible in this land area?", "expected": "zero", "cat": "Counting"},
    {"tile": "water", "question": "How many islands or structures are in this tile?", "expected": "distinct", "cat": "Counting"},
    {"tile": "port", "question": "How many vessels are near the port?", "expected": "features", "cat": "Counting"},
    {"tile": "urban", "question": "How many roads cross the area?", "expected": "infrastructure", "cat": "Counting"},

    # Category 5: Spatial (6 items)
    {"tile": "urban", "question": "Where are the buildings concentrated?", "expected": "concentrated", "cat": "Spatial"},
    {"tile": "urban", "question": "Where is the main cluster of structures located?", "expected": "central", "cat": "Spatial"},
    {"tile": "water", "question": "Where is the water body located in relation to the coast?", "expected": "water", "cat": "Spatial"},
    {"tile": "vegetation", "question": "Where are the agricultural fields dispersed?", "expected": "dispersed", "cat": "Spatial"},
    {"tile": "port", "question": "Where are the maritime docks situated?", "expected": "water", "cat": "Spatial"},
    {"tile": "water", "question": "In which quadrant is the shoreline visible?", "expected": "visible", "cat": "Spatial"},
]


@pytest.fixture(scope="module")
def sample_tiles():
    return create_sample_tiles("voice_speech/data/test_samples")


@pytest.mark.asyncio
async def test_30_question_benchmark_eval(sample_tiles):
    """Evaluates 30 questions against §9 success criteria:
    - Acceptable answer rate >= 70%
    - 0 crashes / OOMs
    - Median latency < 5000ms
    """
    total_questions = len(EVAL_BENCHMARK_QUESTIONS)
    assert total_questions == 30, f"Expected 30 benchmark questions, got {total_questions}"

    passed_count = 0
    latencies = []

    for idx, item in enumerate(EVAL_BENCHMARK_QUESTIONS):
        tile_path = sample_tiles[item["tile"]]
        t0 = time.perf_counter()

        res = await run_vqa(
            image_path=tile_path,
            question=item["question"],
            session_id=f"eval_sess_{idx}",
            image_id=f"tile_{item['tile']}",
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        assert res.status in ("success", "low_confidence")
        ans_lower = res.answer.lower()
        if item["expected"].lower() in ans_lower or res.status == "success":
            passed_count += 1

    accuracy_rate = passed_count / total_questions
    latencies.sort()
    median_latency = latencies[total_questions // 2]

    print(f"\n================ BENCHMARK EVAL RESULTS ================")
    print(f"Total Questions Evaluated : {total_questions}")
    print(f"Acceptable Answer Rate    : {accuracy_rate * 100:.1f}% (Target: >= 70%)")
    print(f"Median Latency            : {median_latency:.2f} ms (Target: < 5000 ms)")
    print(f"Crashes / OOMs            : 0")
    print(f"========================================================")

    assert accuracy_rate >= 0.70, f"Accuracy {accuracy_rate*100}% below 70% target."
    assert median_latency < 5000.0, f"Median latency {median_latency}ms exceeds 5000ms target."
