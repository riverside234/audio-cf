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
- Write one complete, atomic sentence of 5-30 words and end it with punctuation.
- Use facts only from the single target evidence source; never move a fact between AUDIO_N sources.
- SUPPORTED: state one fact explicitly established by that audio's captions. Set contradiction_basis to "none".
- CONTRADICTED: change exactly one explicit caption fact into a mutually incompatible alternative about the same event, object, action, or attribute.
- A contradiction must be proved by positive caption evidence from the same audio. A different or additional event is not necessarily incompatible.
- Never infer that a sound is absent because captions omit it. Never use another audio's captions to prove contradiction.
- supporting_caption_phrases must quote or closely copy the shortest phrases that establish the support or contradiction.
- For CONTRADICTED, contradiction_basis must name the changed proposition, quote the incompatible caption fact, and explain why both cannot be true in the claimed form.
- Do not add unsupported counts, identity, language, demographics, location, emotion, intent, timing, or operator identity.
- forbidden_inferences lists only tempting conclusions that must not be added; use an empty list when none apply.
- Do not mention datasets, files, captions, schemas, or metadata in claim_text.
- Set confidence from 0 to 1 based on how explicitly the selected phrases establish the label.
- If no explicit mutually incompatible modification is available, do not invent one; retry with another caption fact.
- Return JSON only, without markdown or visible reasoning.
