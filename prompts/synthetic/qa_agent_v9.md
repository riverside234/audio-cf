Create one natural benchmark question and concise explanation from this validated claim.

Validated claim:
{claim_record_json}

Retry feedback:
{validation_feedback}

Reasoning policy:
{reasoning_instruction}

Return one JSON object with exactly these keys: question, claim_evaluation_explanation.

Rules:
- Evaluate the literal claim_text. Never substitute an intended edit mentioned only in contradiction_basis or discuss planning and self-correction.
- question: write a standalone grammatical question of at least six words ending with ?. Identify the literal claim unambiguously using the full claim or its distinguishing detail.
- Ask for both the supported/contradicted judgment and which audio or recording determines it. Never name an AUDIO_N label or supply the caption's corrective alternative in the question.
- Vary framing and clause order instead of defaulting to one template.
- claim_evaluation_explanation: briefly connect the literal claim and sole evidence source to the label. For CONTRADICTED, identify at least one central caption conflict; do not imply every modifier conflicts.
- Do not discuss confidence, schemas, or validation metadata, and do not add facts.
- Do not output answer, answer_source, or required_evidence_sources; the application constructs them deterministically.
- Return JSON only, without markdown or visible reasoning.
