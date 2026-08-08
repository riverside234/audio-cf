Create one natural question and one benchmark answer from a validated claim.

Validated claim:
{claim_record_json}

Retry feedback:
{validation_feedback}

Reasoning policy:
{reasoning_instruction}

Return one JSON object with exactly these keys: question, answer, answer_source, claim_evaluation_explanation, required_evidence_sources.

Question rules:
- Write a standalone, grammatical, naturally varied question.
- It may quote, paraphrase, or use only the distinguishing part of claim_text, but must identify the claim unambiguously.
- Ask for both the faithful/counterfactual class and relevant audio source labels.
- Do not repeatedly use the template "Is the claim ... supported or contradicted by ...?"
- Do not reveal the answer. Use at least six words, end with ?, and avoid unfinished quotations or phrases.

Answer rules:
- answer must be an actual JSON/Python-style list, not prose or a quoted list.
- Item 1 must exactly equal claim_type: "faithful" or "counterfactual".
- Later items must exactly equal evidence_sources in order.
- Single source: ["counterfactual", "AUDIO_1"]. Multi-source: ["faithful", "AUDIO_1", "AUDIO_3"].
- answer_source and required_evidence_sources must each be lists exactly equal to evidence_sources.
- Briefly justify the class and sources in claim_evaluation_explanation using only validated evidence.
- Verify that the question is complete and answer equals [claim_type, *evidence_sources].
- Return JSON only, without markdown or visible reasoning.
