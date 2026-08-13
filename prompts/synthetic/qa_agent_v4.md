Create one natural benchmark question and answer from this validated claim.

Validated claim:
{claim_record_json}

Retry feedback:
{validation_feedback}

Reasoning policy:
{reasoning_instruction}

Return one JSON object with exactly these keys: question, answer, answer_source, claim_evaluation_explanation, required_evidence_sources.

Question rules:
- Write a standalone, grammatical, naturally varied question.
- Quote, paraphrase, or use only the distinguishing part of claim_text, but identify it unambiguously.
- Ask for its evidence judgment and determining audio.
- Do not repeatedly use the template "Is the claim ... supported or contradicted by ...?"
- Do not reveal the answer. Use at least six words, end with ?, and avoid unfinished phrases.

Answer rules:
- answer must be an actual two-item JSON/Python-style list, not prose or a quoted list.
- Map claim_status as follows: SUPPORTED -> "supported", CONTRADICTED -> "contradicted", UNSUPPORTED -> "unsupported".
- Item 2 is the sole evidence source for supported/contradicted claims and "NONE" for unsupported claims.
- Use exactly one of these forms: ["supported", "AUDIO_1"], ["contradicted", "AUDIO_2"], or ["unsupported", "NONE"].
- Never put claim_type values such as "faithful" or "counterfactual" in answer.
- For supported/contradicted claims, answer_source and required_evidence_sources each equal the one-item evidence_sources list.
- For unsupported claims, those three source lists are empty; only answer uses "NONE".
- Briefly justify the judgment and source in claim_evaluation_explanation using only validated evidence.
- Verify that the question is complete and answer has exactly two strings.
- Return JSON only, without markdown or visible reasoning.
