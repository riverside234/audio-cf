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
- Use only explicit caption facts; never treat an unmentioned sound as absent.
- Match claim_type, claim_status, evidence_sources, and counterfactual_edit_type to the target.
- A faithful claim is short, atomic, and directly supported.
- A counterfactual must conflict through the requested edit and explain the explicit contradiction_basis.
- supporting_caption_phrases must closely copy short phrases from the captions.
- List any tempting unsupported inference in forbidden_inferences and set confidence from 0 to 1.
- Do not infer counts, order, identity, language, demographics, location, emotion, or intent unless stated.
- Do not mention datasets, files, captions, schemas, or metadata in claim_text.
- Return JSON only, without markdown or visible reasoning.
