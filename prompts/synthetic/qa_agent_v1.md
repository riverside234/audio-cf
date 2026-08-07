You create one training question and answer from a validated caption-grounded claim.

Validated claim:
{claim_record_json}

Retry feedback:
{validation_feedback}

Reasoning policy:
{reasoning_instruction}

Return exactly one JSON object with these keys: question, answer, answer_source, claim_evaluation_explanation, required_evidence_sources.

Rules:
- Ask whether the claim is supported or contradicted by the named audio source or sources.
- State the evaluation clearly and name the correct AUDIO_N source labels.
- required_evidence_sources must exactly equal the claim's evidence_sources.
- Base the explanation only on the validated claim and its supporting or contradicting evidence.
- Keep the question, answer, and explanation concise for later SFT.
- Return JSON only, without markdown or visible reasoning.
