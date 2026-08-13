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
- Evidence may come from one caption or several captions for that same audio.
- When using several captions, select only phrases that converge on and strengthen one atomic proposition. Do not combine independent events into a multi-event claim.
- SUPPORTED: state one atomic fact explicitly established by one or more captions from the target audio. Set contradiction_basis to "none".
- CONTRADICTED: change exactly one atomic proposition established by one or more same-audio captions into a mutually incompatible alternative about the same event, object, action, or attribute.
- A contradiction must be proved by positive caption evidence from the same audio. A different or additional event is not necessarily incompatible.
- Never infer that a sound is absent because captions omit it. Never use another audio's captions to prove contradiction.
- supporting_caption_phrases must list the shortest phrase from every caption used as evidence. Use one phrase when one caption is sufficient and multiple phrases only when they jointly strengthen the same proposition.
- For CONTRADICTED, contradiction_basis must quote every supporting_caption_phrases item, name the changed proposition, and explain why the evidence and claim cannot both be true in the claimed form.
- Do not add unsupported counts, identity, language, demographics, location, emotion, intent, timing, or operator identity.
- forbidden_inferences lists only tempting conclusions that must not be added; use an empty list when none apply.
- Do not mention datasets, files, captions, schemas, or metadata in claim_text.
- Set confidence from 0 to 1 based on how explicitly the selected phrases establish the label.
- If no explicit mutually incompatible modification is available, select another caption fact instead of inventing one.
- Return JSON only, without markdown or visible reasoning.
