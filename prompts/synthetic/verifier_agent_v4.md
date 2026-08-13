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
- Counterfactuals require an explicit contradiction, not caption absence.
- The question must be complete, natural, and ask for both the evidence judgment and the audio source that determines it.
- The answer must contain exactly two strings.
- Map SUPPORTED to ["supported", "AUDIO_N"] and CONTRADICTED to ["contradicted", "AUDIO_N"], using the claim's sole evidence source.
- Map UNSUPPORTED to ["unsupported", "NONE"], with empty evidence_sources, answer_source, and required_evidence_sources.
- Never use faithful or counterfactual as the first answer label.
- answer_source and required_evidence_sources must both exactly equal evidence_sources.
- Reject unsupported counts, order, identity, language, demographics, location, emotion, or intent.
- Return PASS only when every check passes; otherwise return FAIL with concise errors and corrections.
- Return JSON only, without markdown or visible reasoning.
