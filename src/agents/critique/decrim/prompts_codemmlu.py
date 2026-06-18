"""DeCRIM critique prompts — CodeMMLU (multiple-choice) dataset.

CodeMMLU is multiple-choice: each task gives a Python code problem and four candidate
options (A, B, C, D); exactly one is reference-correct. The agent's answer is therefore a
LETTER (plus its reasoning), NOT a code implementation — so the "decompose" step produces a
checklist of verification criteria, and the "critique" step checks the chosen option against
them.

Mirrors the DS-1000 module and exposes the SAME public names (DECOMPOSE_PROMPT,
CRITIQUE_PROMPT, CONSTRAINTS_HEADER), so the agent switches datasets by import path only.

The critique is deliberately aligned with the ReAct SOLVER_PROMPT gating: only concrete,
verifiable reasons (behaviour difference, failed example, wrong boundary, canonical form)
should change the answer — style / naming / theoretical edge cases must be ignored.

Templates go through ``str.format()``; any literal brace must be doubled ({{ }}).
"""

CONSTRAINTS_HEADER = "Verification Criteria:"

DECOMPOSE_PROMPT = """You are helping decide a MULTIPLE-CHOICE question about Python code.
The task gives a problem and four candidate options labelled A, B, C, D; exactly one is the
reference-correct answer. Your job is to extract a STRICT enumerated checklist of concrete,
verifiable criteria that the correct option must satisfy — the things every option should be
checked against.

Base each criterion on the problem itself, e.g.:
- required behaviour on the docstring / examples;
- boundary cases: empty input, a single element, 0, 1, negatives, duplicates, first/last index;
- correct operator / comparison / formula and no off-by-one;
- guards the spec needs (e.g. empty input returns the sentinel BEFORE dividing or indexing);
- when options are functionally equivalent, the canonical / minimal reference form.

FORMAT REQUIREMENTS:
- Use ONLY a numbered list where each line starts with a digit followed immediately by a dot and a space (e.g., "1. ", "2. ", "3. ").
- Each criterion must be on a separate line.
- Do NOT use markdown formatting, parentheses, dashes, Roman numerals, or any other numbering styles.
- Output ONLY the list itself, without introductory text, explanations, or concluding remarks.

Problem (with options A/B/C/D): {question}

Verification Criteria:
"""

CRITIQUE_PROMPT = """You are helping decide a MULTIPLE-CHOICE question about Python code.
You are given the problem (with four options A/B/C/D), a checklist of verification criteria,
and an INITIAL ANSWER from another agent — a single letter (A, B, C, or D), possibly with the
reasoning behind it.

Check the INITIAL ANSWER against each criterion. For each one give a SHORT, concrete reason
and end the line with "Criterion satisfied" or "Criterion not satisfied". A reason must be
verifiable — a behaviour difference, a failed docstring example, a wrong boundary, or (for
functionally equivalent options) which form is canonical. Do NOT raise issues of style,
naming, or theoretical edge cases that do not change which option is reference-correct.

Finish with a single line:
VERDICT: <letter> — <concrete reason>
(confirm the initial letter, or name the letter that is actually reference-correct).

Problem (with options A/B/C/D): {question}

{constraints_text}

Initial Answer: {answer}

Analyze each criterion one by one, then give the VERDICT:
"""
