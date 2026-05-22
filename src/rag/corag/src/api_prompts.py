from typing import List, Dict


def get_generate_api_subquery_prompt(
    query: str,
    past_subqueries: List[str],
    past_subanswers: List[str],
    task_desc: str,
) -> List[Dict]:
    assert len(past_subqueries) == len(past_subanswers)
    past = ""
    for idx in range(len(past_subqueries)):
        past += f"""Intermediate API {idx + 1}: {past_subqueries[idx]}
Intermediate summary {idx + 1}: {past_subanswers[idx]}\n"""
    past = past.strip()

    prompt = f"""You are solving a Python programming problem by iteratively selecting library APIs and reading their documentation. Given the programming task and any APIs already explored, pick the next most useful API to look up.

Allowed libraries (dot-notation only):
- pandas, numpy, matplotlib, sklearn, scipy, torch, tensorflow

Rules:
- Output exactly one API per line in dot-notation, e.g. pandas.DataFrame.groupby
- Do not repeat APIs already listed below
- Prefer specific methods/classes over whole modules
- Do not output explanations, numbering, or any other text

## APIs already explored
{past or "Nothing yet"}

## Task description
{task_desc}

## Programming problem
{query}

Respond with a single API name in dot-notation only."""

    return [{"role": "user", "content": prompt}]


def get_generate_api_intermediate_answer_prompt(
    api: str,
    documents: List[str],
    main_task: str,
) -> List[Dict]:
    context = ""
    for doc in documents:
        context += f"{doc}\n\n"

    prompt = f"""You are preparing a compact API reference for a code-generation model. Using ONLY the documentation below, summarize how `{api}` can help solve the programming problem. Do not invent parameters or behavior not present in the docs.

Include when available:
- Signature and return type (one line)
- Key arguments relevant to this problem
- One minimal usage pattern

If the documentation is unrelated or empty, respond exactly: No relevant information found

## API documentation
{context.strip()}

## Programming problem
{main_task}

## API to summarize
{api}

Respond with a concise summary only, no preamble."""

    return [{"role": "user", "content": prompt}]


def get_generate_api_final_answer_prompt(
    query: str,
    past_subqueries: List[str],
    past_subanswers: List[str],
    task_desc: str,
    documents: list[str] | None = None,
) -> List[Dict]:
    assert len(past_subqueries) == len(past_subanswers)
    past = ""
    for idx in range(len(past_subqueries)):
        past += f"""Intermediate API {idx + 1}: {past_subqueries[idx]}
Intermediate summary {idx + 1}: {past_subanswers[idx]}\n"""
    past = past.strip()

    context = ""
    if documents:
        for idx, doc in enumerate(documents):
            context += f"Doc {idx}: {doc}\n\n"

    prompt = f"""Given API documentation summaries gathered while exploring APIs for a programming problem, write the solution code. Use only information supported by the summaries and docs. Intermediate summaries may be incomplete.

## Additional documentation
{context.strip() or "None"}

## Explored APIs and summaries
{past or "Nothing yet"}

## Task description
{task_desc}

## Programming problem
{query}

Write a short solution following the required format. Place executable code between <code> and </code> tags. Do not add explanations outside the tags."""

    return [{"role": "user", "content": prompt}]
