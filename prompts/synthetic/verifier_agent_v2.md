You strictly verify one caption-grounded training example. Do not rewrite it creatively.

Relevant captions:
{audio_context}

Target:
{target_condition_json}

Claim:
{claim_record_json}

Question and answer:
{qa_record_json}

Retry feedback:
{validation_feedback}

Reasoning policy:
{reasoning_instruction}

Return exactly one JSON object with these keys: verifier_status, validation_errors, validation_notes, corrected_claim_status, corrected_evidence_sources.

Checks:
- Claim facts and status must follow explicit captions and the target.
- Source labels in the claim and QA must be valid and consistent.
- Counterfactuals require explicit contradiction, not caption absence.
- The QA must evaluate the claim, state supported or contradicted, and name the correct sources.
- Reject unsupported counts, order, identity, language, demographics, location, emotion, or intent.
- Return PASS only when every check passes; otherwise return FAIL with concise errors and corrections.
- Return JSON only, without markdown or visible reasoning.
