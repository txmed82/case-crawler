from __future__ import annotations

DIFFICULTY_RULES = {
    "medical_student": {
        "vignette": "Include most key findings with fewer distractors. Classic textbook presentation.",
        "decision_tree": "Provide 2-3 choices. One clearly correct, one common mistake, one possible catastrophic error.",
        "complications": "Simple cause-effect relationships.",
        "knowledge": "Foundational pathophysiology. Basic diagnostic and treatment knowledge.",
        "phase_count_min": 3,
        "phase_count_max": 3,
        "decisions_per_phase": "2-3",
        "quality_distribution": "1 optimal, 1 harmful/catastrophic, 0-1 suboptimal",
    },
    "resident": {
        "vignette": "Some findings missing, moderate distractors. Atypical or overlapping presentations.",
        "decision_tree": "Provide 3-4 choices with nuanced distinctions. Include management algorithm decisions.",
        "complications": "Multi-step cascades where one error leads to another.",
        "knowledge": "Management algorithms, workup prioritization, time-sensitive decisions.",
        "phase_count_min": 5,
        "phase_count_max": 5,
        "decisions_per_phase": "3-4",
        "quality_distribution": "1 optimal, 1-2 acceptable/suboptimal, 1 harmful",
    },
    "attending": {
        "vignette": "Incomplete data, many distractors, red herrings. Rare variants or multiple concurrent pathologies.",
        "decision_tree": "Provide 4-5 choices with subtle distinctions between reasonable options.",
        "complications": "System-level failures (e.g., missed consult leads to delayed OR leads to herniation).",
        "knowledge": "Nuanced judgment calls, resource constraints, ambiguity tolerance.",
        "phase_count_min": 7,
        "phase_count_max": 10,
        "decisions_per_phase": "4-5",
        "quality_distribution": "1 optimal, 2 acceptable, 1-2 suboptimal, 0-1 catastrophic",
    },
}

CASE_GENERATOR_SYSTEM = """You are a clinical case author creating realistic, decision-forcing medical scenarios.

Your cases must:
- Be messy and incomplete, like real medicine
- Force the clinician to make decisions with incomplete information
- Include distractors that could lead to wrong diagnoses
- Be grounded in real medical knowledge from the provided sources
- Have diverse patient demographics (vary age, sex, background)
- Avoid demographic stereotypes (do not default to stereotypical presentations)

You will receive:
1. A medical topic
2. A difficulty level with specific rules
3. Retrieved medical knowledge from real sources

Generate a realistic clinical vignette with patient demographics and ground truth."""

DECISION_TREE_SYSTEM = """You are a clinical decision tree architect.

Given a clinical vignette and its ground truth, build a decision tree that:
- Has exactly ONE correct path
- Has plausible wrong paths that a clinician might actually choose
- Labels each wrong path as "common_mistake" or "catastrophic"
- Provides realistic consequences for each wrong choice
- Includes a complications layer showing what happens with delayed or incorrect decisions

Each choice must include:
- The action the clinician would take
- Whether it is correct
- Clinical reasoning for why someone might choose it
- The realistic outcome of that choice
- For wrong choices: the consequence and what error type it represents

Also generate complications that show temporal consequences:
- What happens if diagnosis is delayed
- What happens if incorrect treatment is given"""

CLINICAL_REVIEWER_SYSTEM = """You are a senior clinical reviewer evaluating AI-generated medical cases for quality.

Score each case on three dimensions (0.0 to 1.0):

**Accuracy (0.0-1.0):**
- Is the diagnosis correct for the presented findings?
- Are the treatment options real and appropriate?
- Are lab values, vital signs, and timelines physiologically possible?
- Do the decision tree outcomes match known clinical evidence?

**Pedagogy (0.0-1.0):**
- Is the case appropriately challenging for the stated difficulty level?
- Are distractors plausible but distinguishable with proper knowledge?
- Does the decision tree cover the most important learning points?
- Would a learner gain meaningful clinical reasoning from this case?

**Bias (0.0-1.0):**
- Does the case avoid demographic stereotyping?
- Is the patient presentation free from cultural assumptions?
- Would the case work equally well with a different patient demographic?
- Are gendered language patterns avoided?

If ANY score is below the threshold, you MUST reject and provide specific, actionable feedback.
Your feedback should tell the generator exactly what to fix."""


def build_case_generator_prompt(topic: str, difficulty: str, context: str) -> str:
    rules = DIFFICULTY_RULES.get(difficulty, DIFFICULTY_RULES["resident"])
    return f"""Generate a clinical case for the following topic.

## Topic
{topic}

## Difficulty Level: {difficulty}
- Vignette: {rules['vignette']}
- Knowledge level: {rules['knowledge']}

## Medical Knowledge (from real sources)
{context}

## Instructions
Create a realistic clinical vignette with:
1. Patient demographics (age, sex, relevant background)
2. The clinical presentation (history, exam findings, initial labs if relevant)
3. A decision prompt ("What would you do next?")
4. Ground truth: the correct diagnosis, optimal next step, rationale, and key findings

Make the vignette realistic and messy — like a real patient encounter, not a textbook question."""


def build_decision_tree_prompt(vignette: str, ground_truth_json: str, difficulty: str, context: str) -> str:
    rules = DIFFICULTY_RULES.get(difficulty, DIFFICULTY_RULES["resident"])
    return f"""Build a decision tree for this clinical case.

## Vignette
{vignette}

## Ground Truth
{ground_truth_json}

## Difficulty Level: {difficulty}
- Decision tree: {rules['decision_tree']}
- Complications: {rules['complications']}

## Medical Knowledge (from real sources)
{context}

## Instructions
Create:
1. Decision choices — one correct path and plausible wrong paths
2. Complications — what happens with delayed diagnosis or incorrect treatment

Each wrong choice must have an error_type of "common_mistake" or "catastrophic"."""


def build_reviewer_prompt(case_json: str, context: str, threshold: float) -> str:
    return f"""Review this AI-generated clinical case for quality.

## Generated Case
{case_json}

## Source Material (what the case was built from)
{context}

## Scoring Threshold
All scores must be >= {threshold} for approval.

## Instructions
Score accuracy, pedagogy, and bias (0.0-1.0 each).
Set approved=true only if ALL scores meet the threshold.
If rejecting, provide specific, actionable notes explaining what to fix."""


def build_retry_prompt(original_prompt: str, reviewer_notes: list[str]) -> str:
    notes_text = "\n".join(f"- {note}" for note in reviewer_notes)
    return f"""{original_prompt}

## IMPORTANT: Previous Attempt Was Rejected
The clinical reviewer provided this feedback. You MUST address each issue:

{notes_text}

Fix these specific issues while preserving what was already correct."""


CASE_PLANNER_SYSTEM = """You are a clinical case planner creating structured blueprints for multi-step medical simulations.

Your blueprints must:
- Define a realistic clinical arc from presentation through management
- Specify the exact number of phases appropriate for the difficulty level
- Place decision points at genuine clinical crossroads
- Include branching points where wrong decisions lead to different outcomes
- Anticipate realistic complications that could arise
- Ground all clinical reasoning in the provided source material

You produce STRUCTURE ONLY — no prose vignettes, no lab values, no imaging reports.
Each phase specifies: what's happening clinically, what diagnostics are available/pending,
what type of decision is being made, and what the correct action is with reasoning."""

BLUEPRINT_REVIEWER_SYSTEM = """You are a clinical blueprint reviewer validating the structural soundness of a multi-step case plan.

Evaluate the blueprint on four criteria:
1. Is the clinical arc medically sound? Does the diagnosis match the planned workup and management?
2. Do phases follow a logical temporal sequence? Are time offsets realistic?
3. Are branching points at genuine decision points where a clinician could reasonably go wrong?
4. Is the phase count appropriate for the stated difficulty level?

If ANY criterion fails, reject with specific feedback. Your feedback should tell the planner
exactly what to fix. Set approved=true only if all criteria pass."""

PHASE_RENDERER_SYSTEM = """You are a clinical case phase renderer. Given a case blueprint and a specific phase skeleton,
you generate the detailed clinical content for that phase.

You must produce:
1. A realistic narrative for this moment in the case
2. Structured vital signs (if clinically relevant at this phase)
3. Structured lab results using EXACT units and reference ranges from the provided templates
4. Structured imaging results using modality-appropriate terminology
5. Decision options graded on a 5-level scale: optimal, acceptable, suboptimal, harmful, catastrophic
6. A phase outcome describing what happens next on the optimal path

Rules:
- Lab values must use the exact unit and precision from the panel template
- Abnormal values must be physiologically consistent with the diagnosis
- Flag any value outside reference range (H/L) and any critical value as "critical"
- Imaging findings must use terminology appropriate for the modality (no "hyperdense" for MRI)
- Vital signs must be physiologically coherent
- Decision quality grades must have clear clinical reasoning
- Every phase must have exactly ONE optimal decision"""

CONSISTENCY_CHECKER_SYSTEM = """You are a clinical consistency checker reviewing a set of rendered case phases for cross-phase coherence.

Check for:
1. Vital sign continuity — trends must be physiologically logical. A patient with HR 120 should not
   suddenly have HR 68 without treatment explanation.
2. Lab value coherence — trending values (troponin, lactate, Hgb) must move in clinically plausible
   directions across phases.
3. Temporal logic — order-to-result times must be realistic (stat CBC ~30min, cultures ~48h, CT ~30-45min).
   A test ordered at T+10min cannot result at T+5min.
4. Narrative continuity — patient state must not contradict across phases. If intubated in phase 4,
   cannot be conversational in phase 5.
5. Decision coherence — available actions must not include things already done in prior phases.

For each issue found, report: phase_number, field, issue description, suggested fix.
If no issues found, return an empty list."""


def build_case_planner_prompt(topic: str, difficulty: str, context: str) -> str:
    rules = DIFFICULTY_RULES.get(difficulty, DIFFICULTY_RULES["resident"])
    return f"""Create a case blueprint for the following topic.

## Topic
{topic}

## Difficulty Level: {difficulty}
- Vignette style: {rules['vignette']}
- Knowledge level: {rules['knowledge']}
- Phase count: {rules['phase_count_min']}-{rules['phase_count_max']} phases
- Decisions per phase: {rules['decisions_per_phase']}
- Quality distribution: {rules['quality_distribution']}
- Complications: {rules['complications']}

## Medical Knowledge (from real sources)
{context}

## Instructions
Create a structured blueprint with:
1. The diagnosis and clinical arc
2. Phase-by-phase plan with time offsets, decision types, and correct actions
3. Branching points where wrong decisions cause different outcomes (redirect, fork, or terminal)
4. Expected complications that could arise during the case

Do NOT generate prose, lab values, or imaging reports — structure only."""


def build_blueprint_reviewer_prompt(blueprint_json: str, context: str) -> str:
    return f"""Review this case blueprint for structural soundness.

## Blueprint
{blueprint_json}

## Source Material
{context}

## Instructions
Evaluate: clinical arc validity, temporal sequence logic, branching point placement,
and phase count appropriateness. Set approved=true only if all criteria pass.
If rejecting, provide specific actionable feedback."""


def build_phase_renderer_prompt(
    blueprint_json: str,
    phase_json: str,
    difficulty: str,
    context: str,
    lab_panel_context: str,
) -> str:
    rules = DIFFICULTY_RULES.get(difficulty, DIFFICULTY_RULES["resident"])
    return f"""Render the detailed clinical content for this phase.

## Case Blueprint
{blueprint_json}

## Phase to Render
{phase_json}

## Difficulty Level: {difficulty}
- Decisions per phase: {rules['decisions_per_phase']}
- Quality distribution: {rules['quality_distribution']}
- Decision tree style: {rules['decision_tree']}

## Medical Knowledge (from real sources)
{context}

## Lab Panel Reference Ranges
{lab_panel_context}

## Instructions
Generate:
1. Narrative text for this clinical moment
2. Vital signs (if relevant)
3. Lab results using EXACT units and ranges from the panel reference above
4. Imaging results using modality-appropriate terminology
5. Decision options with 5-level quality grading
6. Phase outcome (next phase, patient status, transition narrative)"""


def build_consistency_checker_prompt(phases_json: str) -> str:
    return f"""Check these rendered case phases for cross-phase consistency.

## Rendered Phases
{phases_json}

## Instructions
Check vital sign continuity, lab value coherence, temporal logic,
narrative continuity, and decision coherence.

Return a list of issues found. Each issue should have:
- phase_number: which phase has the problem
- field: which field is inconsistent (e.g., "vitals.hr", "lab_results.Hgb")
- issue: what the inconsistency is
- suggested_fix: how to resolve it

If no issues found, return an empty list."""
