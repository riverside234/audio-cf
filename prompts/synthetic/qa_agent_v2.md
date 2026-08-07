You create one training question and answer from a validated caption-grounded claim.

Validated claim:
{claim_record_json}

Retry feedback:
{validation_feedback}

Reasoning policy:
{reasoning_instruction}

Return exactly one JSON object with these keys: question, answer, answer_source, claim_evaluation_explanation, required_evidence_sources.

Rules:
- Write a standalone, grammatical question that includes the complete claim and relevant AUDIO_N source labels.
- Ask whether the claim is supported, contradicted, or partially supported by the named audio evidence.
- Prefer: Is the claim "<complete claim_text>" supported or contradicted by AUDIO_N?
- Never truncate the claim, leave a quotation unfinished, or end with a dangling phrase such as "Is the claim".
- The question must contain at least eight words and end with a question mark.
- State the evaluation clearly in the answer and name the correct AUDIO_N source labels.
- required_evidence_sources must exactly equal the claim's evidence_sources.
- Base the explanation only on the validated claim and its evidence.
- Keep the question, answer, and explanation concise for later SFT.
- Before returning, verify that every string is complete and the question is understandable without additional context.
- Return JSON only, without markdown or visible reasoning.
