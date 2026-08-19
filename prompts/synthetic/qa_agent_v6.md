Create one natural benchmark question and answer from this validated claim.

Validated claim:
{claim_record_json}

Retry feedback:
{validation_feedback}

Reasoning policy:
{reasoning_instruction}

Return one JSON object with exactly these keys: question, answer, answer_source, claim_evaluation_explanation, required_evidence_sources.

Rules:
- question: write a standalone grammatical question of at least six words ending with ?. Identify the claim unambiguously using the full claim or only its distinguishing detail.
- Ask for both the supported/contradicted judgment and the determining audio without revealing either answer.
- Vary framing and clause order: lead with the event, evidence judgment, or source request instead of defaulting to one template.
- answer: map SUPPORTED to ["supported", "AUDIO_N"] and CONTRADICTED to ["contradicted", "AUDIO_N"], replacing AUDIO_N with the claim's sole evidence source.
- answer_source and required_evidence_sources must each exactly equal the claim's one-item evidence_sources list.
- claim_evaluation_explanation: briefly connect the label and source to the validated claim evidence without adding facts.
- Return JSON only, without markdown or visible reasoning.
