Create one caption-grounded claim for a supported-versus-contradicted benchmark.

Relevant captions:
{audio_context}

Target:
{target_condition_json}

Retry feedback:
{validation_feedback}

Reasoning policy:
{reasoning_instruction}

Return one JSON object with exactly these keys: claim_text, claim_type, claim_status, evidence_sources, counterfactual_edit_type, supporting_caption_phrases, contradiction_basis, forbidden_inferences, confidence.

Rules:
- Copy claim_type, claim_status, evidence_sources, and counterfactual_edit_type exactly from the target.
- claim_text: write one complete 5-30 word sentence about one coherent event in the target audio. It may combine related propositions, never independent events.
- Use only facts from the target audio. Select the shortest exact or near-exact supporting caption phrases; multiple phrases must describe the same event.
- SUPPORTED: every claimed proposition must be explicit in the selected evidence. Set contradiction_basis to "none".
- CONTRADICTED: replace one or more caption-established facts with clearly incompatible alternatives. Positive evidence from the target audio must disprove every changed proposition.
- Explicit relative or subjective contrasts such as loud versus quiet or gentle versus aggressive are valid when the caption states the grounding attribute and ordinary meanings conflict.
- Caption omission, general knowledge, another audio, or a merely different or additional event never proves contradiction. Select another caption fact if necessary.
- For CONTRADICTED, contradiction_basis must quote every selected phrase, identify each change, and explain the incompatibility.
- Do not add unsupported counts, identity, language, demographics, location, emotion, intent, timing, or operator identity.
- forbidden_inferences lists only tempting details that must not be added; use an empty list when none apply.
- Keep claim_text free of references to captions, datasets, files, schemas, or metadata.
- Set confidence from 0 to 1 according to how explicitly the selected evidence establishes the label.
- Return JSON only, without markdown or visible reasoning.
