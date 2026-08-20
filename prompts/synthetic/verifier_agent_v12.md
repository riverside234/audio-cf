Verify one caption-grounded benchmark example without rewriting it. This is the mandatory final acceptance gate.

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
- Only SUPPORTED and CONTRADICTED are valid final labels, with exactly one evidence audio.
- The claim must describe one coherent scene; it may contain one proposition or several related propositions.
- SUPPORTED requires every claimed proposition to be explicit in one or more captions from the evidence audio.
- CONTRADICTED requires at least one central claim detail to conflict with positive captions from that audio. Other details may be supported or neutral and do not each need separate contradictory evidence.
- Treat contradiction as disagreement with the reference captions, not proof of physical impossibility. Same-scene object, source, tool, action, direction, count, or explicitly stated attribute substitutions may pass even if the events could theoretically co-occur.
- Reject absence-only reasoning, unrelated additions, general-knowledge-only conflicts, and cross-audio source swaps.
- Reject a contradicted record when a source caption directly states its central claim, or when contradiction_basis describes a change missing from the literal claim_text.
- supporting_caption_phrases must come from the evidence audio. For CONTRADICTED, contradiction_basis must identify at least one central conflict without planning or self-correction.
- FAIL if the question omits the evidence judgment. Asking which audio or recording determines it is optional because the downstream evaluation prompt may request the source separately.
- Whether or not the question requests the source, FAIL if it names an AUDIO_N label or supplies the caption's corrective alternative.
- The explanation must discuss an actual conflict present in the literal claim, not an intended edit found only in contradiction_basis.
- The answer must be exactly ["supported", "AUDIO_N"] or ["contradicted", "AUDIO_N"], matching claim_status and the evidence source. answer_source and required_evidence_sources must equal evidence_sources.
- Return PASS only when every check passes. For PASS, use an empty validation_errors list, null corrected_claim_status, and an empty corrected_evidence_sources list.
- For FAIL, include at least one concise validation error. Supply corrected fields only when the correction is unambiguous.
- Return JSON only, without markdown or visible reasoning.
