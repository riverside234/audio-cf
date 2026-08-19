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
- The claim must describe one coherent event from the sole evidence audio; it may contain one proposition or several related propositions about that event.
- SUPPORTED requires every claimed proposition to be explicitly established by one or more captions from that audio.
- CONTRADICTED requires positive captions from that audio to prove every changed proposition clearly incompatible with the claim.
- Explicitly captioned relative or subjective attributes are valid evidence, including loud versus quiet or gentle versus aggressive, when the alternatives are incompatible in ordinary meaning.
- Reject attributes inferred from general knowledge or caption omission rather than stated caption evidence.
- supporting_caption_phrases must come from the sole evidence audio and describe the same coherent event. contradiction_basis may paraphrase them but must identify every changed fact and explain the incompatibility.
- Caption omission, an unrelated event, or a fact from another audio does not prove contradiction. Reject every cross-audio source swap.
- The question must be complete, natural, and ask for the evidence judgment and determining audio.
- The answer must be exactly ["supported", "AUDIO_N"] or ["contradicted", "AUDIO_N"], matching claim_status and the sole evidence source.
- answer_source and required_evidence_sources must exactly equal evidence_sources.
- Reject unsupported details involving counts, order, identity, language, demographics, location, emotion, intent, or timing.
- Return PASS only when every check passes; otherwise return FAIL with concise errors and corrections.
- Return JSON only, without markdown or visible reasoning.
