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
- Use only explicit caption facts. Never treat an unmentioned sound as absent or add specificity from outside knowledge.
- For a single-source faithful target, use one atomic event or attribute rather than combining several caption facts.
- For a multi-source faithful target, include only one necessary explicit fact from each required source.
- For a counterfactual, change only the dimension requested by counterfactual_edit_type; keep the remaining details grounded.
- For source_swap, AUDIO_N labels are allowed in claim_text. Explain that a fact evidenced for one source was assigned to another; do not claim this proves the sound is absent from the other source.
- For event_swap, attribute_swap, or explicit_fact_modification, identify the original caption fact and the exact changed element in contradiction_basis.
- For false_conjunction, explain the incorrect cross-source attribution without using caption silence as proof of acoustic absence.
- supporting_caption_phrases must quote or closely copy only the shortest caption phrases needed from the evidence sources.
- For a SUPPORTED faithful claim, set contradiction_basis exactly to "none".
- For a CONTRADICTED counterfactual, contradiction_basis must be specific and nonempty.
- forbidden_inferences should list only plausible extra conclusions that must not be added; use an empty list when none are relevant.
- Do not infer counts, order, identity, language, demographics, location, emotion, intent, or operator identity unless stated.
- Do not mention datasets, files, captions, schemas, or metadata in claim_text, except required AUDIO_N labels for a source swap.
- Set confidence from 0 to 1 based on how directly the cited phrases establish the intended claim or contradiction.
- Before returning, verify that claim_text is complete, atomic, and consistent with every target field.
- Return JSON only, without markdown or visible reasoning.
