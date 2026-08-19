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
- Write one complete sentence of 5-30 words about one coherent event in the target audio and end it with punctuation.
- The sentence may contain one proposition or several related propositions about that event. Do not combine independent events.
- Use facts only from the single target evidence source; never move a fact between AUDIO_N sources.
- Use one caption or several captions from that audio, all describing the same coherent event and its related facts.
- SUPPORTED: state one or more related facts explicitly established by captions from the target audio. Set contradiction_basis to "none".
- CONTRADICTED: change one or more related caption-established propositions into objectively incompatible alternatives about the same event.
- Prefer objective changes to event or action occurrence, object or action identity, explicit count, direction, temporal order, or physical state.
- Do not use subjective intensity, manner, emotion, or quality contrasts as the deciding contradiction, including gentle versus aggressive, loud versus quiet, or pleasant versus harsh.
- If the claim changes several propositions, positive caption evidence must independently prove every changed proposition incompatible.
- A contradiction must be proved by positive caption evidence from the same audio. A different or additional event is not necessarily incompatible.
- Never infer that a sound is absent because captions omit it. Never use another audio's captions to prove contradiction.
- supporting_caption_phrases must list the shortest phrase from every caption used as evidence. Use multiple phrases only when they strengthen the same event and its related propositions.
- For CONTRADICTED, contradiction_basis must quote every supporting_caption_phrases item, name every changed proposition, and explain why the evidence and claim cannot both be true in the claimed form.
- Do not add unsupported counts, identity, language, demographics, location, emotion, intent, timing, or operator identity.
- forbidden_inferences lists only tempting conclusions that must not be added; use an empty list when none apply.
- Do not mention datasets, files, captions, schemas, or metadata in claim_text.
- Set confidence from 0 to 1 based on how explicitly the selected phrases establish the label.
- If no objective, explicitly incompatible modification is available, select another caption fact instead of inventing one.
- Return JSON only, without markdown or visible reasoning.
