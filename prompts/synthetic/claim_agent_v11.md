Create one caption-grounded claim for a supported-versus-contradicted benchmark.

Relevant captions:
{audio_context}

Target:
{target_condition_json}

Retry feedback:
{validation_feedback}

Reasoning policy:
{reasoning_instruction}

Return one JSON object with exactly these keys: claim_text, claim_status, evidence_sources, supporting_caption_phrases, contradiction_basis, forbidden_inferences, confidence.

Rules:
- Copy claim_status and evidence_sources exactly from the target.
- claim_text: write one complete 5-30 word sentence about one coherent event in the target audio. It may combine related propositions, never independent events.
- Use only the target audio. Select the shortest exact or near-exact supporting caption phrases; multiple phrases must describe the same scene.
- SUPPORTED: every claimed proposition must be explicit in the selected evidence. Set contradiction_basis to "none".
- CONTRADICTED: change at least one central caption-established fact into a conflicting alternative about the same scene. Valid changes include object, source, tool, action, direction, count, and explicitly stated attribute substitutions.
- Contradiction is judged against the reference captions, not by proving that the alternative was physically absent from the waveform. Alternatives may theoretically co-occur, but the claim must present the changed detail as its description of the captioned scene.
- Other claim details may be caption-supported or neutral; they do not each need separate contradictory evidence.
- Do not use caption omission alone, an unrelated event, general knowledge alone, or another audio as the contradiction. Never return a contradicted claim whose central proposition is directly stated by a selected caption.
- For CONTRADICTED, contradiction_basis must identify at least one central changed fact and its conflicting caption evidence. Ensure the described change appears in the final claim_text; do not include planning or self-correction.
- Explicit caption-grounded contrasts such as loud versus quiet or gentle versus aggressive are valid.
- Do not add unsupported identity, language, demographics, emotion, intent, timing, or operator identity.
- forbidden_inferences lists only tempting details that must not be added; use an empty list when none apply.
- Keep claim_text free of references to captions, datasets, files, schemas, or metadata.
- Set confidence from 0 to 1 according to how clearly the selected evidence establishes the label.
- Return JSON only, without markdown or visible reasoning.
