Create one natural benchmark question and concise explanation from this validated claim.

Validated claim:
{claim_record_json}

Retry feedback:
{validation_feedback}

Reasoning policy:
{reasoning_instruction}

Return one JSON object with exactly these keys: question, claim_evaluation_explanation.

Rules:
- question: write a standalone grammatical question of at least six words ending with ?. Identify the claim unambiguously using the full claim or only its distinguishing detail.
- Ask for both the supported/contradicted judgment and the determining audio without revealing either answer.
- Vary framing and clause order: lead with the event, evidence judgment, or source request instead of defaulting to one template.
- claim_evaluation_explanation: briefly connect the claim's label and sole evidence source to its validated evidence without adding facts.
- Do not output answer, answer_source, or required_evidence_sources; the application constructs them deterministically from the validated claim.
- Return JSON only, without markdown or visible reasoning.
