You create one caption-grounded audio claim.

Relevant captions:
{audio_context}

Target:
{target_condition_json}

Retry feedback:
{validation_feedback}

Reasoning policy:
{reasoning_instruction}

Return exactly one JSON object with these keys: claim_text, claim_type, claim_status, evidence_sources, counterfactual_edit_type, supporting_caption_phrases, contradiction_basis, forbidden_inferences, confidence.

Rules:
- Copy claim_type, claim_status, evidence_sources, and counterfactual_edit_type exactly from the target.
- Write one complete grammatical sentence of 5-30 words for claim_text and end it with punctuation.
- Never infer counts, identity, language, demographics, location, emotion, intent, or operator identity unless stated.
- For a faithful target, use one atomic event or attribute explicitly supported by its sole evidence source.
- For a contradicted counterfactual, change only the requested dimension and keep the remaining details grounded.
- For source_swap, AUDIO_N labels are allowed in claim_text. Explain that a fact evidenced for one source was assigned to another; do not treat caption silence as proof of acoustic absence.
- For event_swap, attribute_swap, or explicit_fact_modification, identify the original caption fact and exact changed element in contradiction_basis.
- For unsupported_detail, add exactly one plausible concrete detail that no caption supports or explicitly contradicts. Set evidence_sources and supporting_caption_phrases to empty lists and contradiction_basis to "none".
- supporting_caption_phrases must quote or closely copy only the shortest phrases needed from evidence_sources.
- For SUPPORTED, set contradiction_basis to "none". For CONTRADICTED, make it specific and nonempty.
- For UNSUPPORTED, do not claim the added detail is false or absent from the audio; only state that it lacks caption evidence.
- forbidden_inferences should list only tempting extra conclusions; use an empty list when none are relevant.
- Do not mention datasets, files, captions, schemas, or metadata in claim_text, except required AUDIO_N labels for source_swap.
- Set confidence from 0 to 1 based on how clearly the captions establish the requested evidence judgment.
- Verify that claim_text is complete, atomic, and consistent with every target field.
- Return JSON only, without markdown or visible reasoning.
