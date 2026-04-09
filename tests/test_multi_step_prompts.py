from casecrawler.generation.prompts import (
    BLUEPRINT_REVIEWER_SYSTEM,
    CASE_PLANNER_SYSTEM,
    CONSISTENCY_CHECKER_SYSTEM,
    DIFFICULTY_RULES,
    PHASE_RENDERER_SYSTEM,
    build_blueprint_reviewer_prompt,
    build_case_planner_prompt,
    build_consistency_checker_prompt,
    build_phase_renderer_prompt,
)

def test_difficulty_rules_have_phase_counts():
    for level in ["medical_student", "resident", "attending"]:
        assert "phase_count_min" in DIFFICULTY_RULES[level]
        assert "phase_count_max" in DIFFICULTY_RULES[level]
    assert DIFFICULTY_RULES["medical_student"]["phase_count_min"] == 3
    assert DIFFICULTY_RULES["medical_student"]["phase_count_max"] == 3
    assert DIFFICULTY_RULES["resident"]["phase_count_min"] == 5
    assert DIFFICULTY_RULES["resident"]["phase_count_max"] == 5
    assert DIFFICULTY_RULES["attending"]["phase_count_min"] == 7
    assert DIFFICULTY_RULES["attending"]["phase_count_max"] == 10

def test_case_planner_system_prompt_exists():
    assert "clinical case planner" in CASE_PLANNER_SYSTEM.lower() or "blueprint" in CASE_PLANNER_SYSTEM.lower()

def test_build_case_planner_prompt():
    prompt = build_case_planner_prompt(topic="SAH", difficulty="resident", context="[Source 1] SAH content")
    assert "SAH" in prompt
    assert "resident" in prompt
    assert "SAH content" in prompt

def test_build_blueprint_reviewer_prompt():
    prompt = build_blueprint_reviewer_prompt(blueprint_json='{"diagnosis": "SAH"}', context="[Source 1] SAH content")
    assert "SAH" in prompt

def test_build_phase_renderer_prompt():
    prompt = build_phase_renderer_prompt(blueprint_json='{"diagnosis": "SAH"}', phase_json='{"phase_number": 1}', difficulty="resident", context="[Source 1] SAH content", lab_panel_context="CBC: WBC 4.5-11.0 K/uL")
    assert "SAH" in prompt
    assert "CBC" in prompt

def test_build_consistency_checker_prompt():
    prompt = build_consistency_checker_prompt(phases_json='[{"phase_number": 1}, {"phase_number": 2}]')
    assert "phase_number" in prompt

def test_phase_renderer_system_prompt_exists():
    assert len(PHASE_RENDERER_SYSTEM) > 100

def test_consistency_checker_system_prompt_exists():
    assert len(CONSISTENCY_CHECKER_SYSTEM) > 100

def test_blueprint_reviewer_system_prompt_exists():
    assert len(BLUEPRINT_REVIEWER_SYSTEM) > 100
