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
- Ask whether the claim is supported or contradicted and which audio supplies the determining evidence.
- Do not repeatedly use the template "Is the claim ... supported or contradicted by ...?"
- Do not reveal the answer. Use at least six words, end with ?, and avoid unfinished phrases.

Answer rules:
- answer must be an actual two-item JSON/Python-style list, not prose or a quoted list.
- Map SUPPORTED to ["supported", "AUDIO_N"] and CONTRADICTED to ["contradicted", "AUDIO_N"].
- Item 2 must exactly equal the claim's sole evidence source.
- Never use "unsupported", "NONE", "faithful", or "counterfactual" as answer labels.
- answer_source and required_evidence_sources must each exactly equal the one-item evidence_sources list.
- Briefly justify the label and source using only the validated claim evidence.
- Verify that the question is complete and answer has exactly two strings.
- Return JSON only, without markdown or visible reasoning.
