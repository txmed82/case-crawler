from __future__ import annotations

DIFFICULTY_RULES = {
    "medical_student": {
        "vignette": "Include most key findings with fewer distractors. Classic textbook presentation.",
        "decision_tree": "Provide 2-3 choices. One clearly correct, one common mistake, one possible catastrophic error.",
        "complications": "Simple cause-effect relationships.",
        "knowledge": "Foundational pathophysiology. Basic diagnostic and treatment knowledge.",
    },
    "resident": {
        "vignette": "Some findings missing, moderate distractors. Atypical or overlapping presentations.",
        "decision_tree": "Provide 3-4 choices with nuanced distinctions. Include management algorithm decisions.",
        "complications": "Multi-step cascades where one error leads to another.",
        "knowledge": "Management algorithms, workup prioritization, time-sensitive decisions.",
    },
    "attending": {
        "vignette": "Incomplete data, many distractors, red herrings. Rare variants or multiple concurrent pathologies.",
        "decision_tree": "Provide 4-5 choices with subtle distinctions between reasonable options.",
        "complications": "System-level failures (e.g., missed consult leads to delayed OR leads to herniation).",
        "knowledge": "Nuanced judgment calls, resource constraints, ambiguity tolerance.",
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
