Strictly verify one caption-grounded benchmark example. Do not rewrite it creatively.

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
- Only SUPPORTED and CONTRADICTED are valid final labels.
- SUPPORTED requires an explicit caption fact from the sole evidence audio.
- CONTRADICTED requires positive caption evidence from that same audio that is mutually incompatible with the claim.
- Caption omission, an unrelated event, or a fact from another audio does not prove contradiction.
- Reject every cross-audio source swap.
- The question must be complete, natural, and ask for the evidence judgment and determining audio.
- The answer must be exactly ["supported", "AUDIO_N"] or ["contradicted", "AUDIO_N"], matching claim_status and the sole evidence source.
- answer_source and required_evidence_sources must exactly equal evidence_sources.
- Reject unsupported details involving counts, order, identity, language, demographics, location, emotion, intent, or timing.
- Return PASS only when every check passes; otherwise return FAIL with concise errors and corrections.
- Return JSON only, without markdown or visible reasoning.
