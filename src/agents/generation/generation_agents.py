from openai import OpenAI
from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text


class QueryGenerator(Agent):
    def __init__(self, url: str, model_name: str, options: int = 3):
        super().__init__("query_generator")
        self.dummy_mode = not (url and model_name)
        self.options = options

        if not self.dummy_mode:
            self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, task: Text) -> Text:
        if self.dummy_mode:
            return task

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": f"""You are search query generator. Using the task description, generate the required number of different search queries to obtain information that will help solve the problem.
                [OUTPUT FORMAT]:
                Yours answer will be divided by lines and each line would be considered as search query and will be passed straight to the search engine.
                So yours answer should consist of just {self.options} lines, each one containing search query. Do not write redundant phrases like "Here are search queries to help you solve the problem". 
                I expect a response in this format containing only search queries:
                
                <query 1>
                <query 2>
                ...
                 """},
                {"role": "user", "content": f"""[PROBLEM DESCRIPTION]:
                {task}

                Provide {self.options} search queries!
                """}
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.split('\n')


_SYSTEM_PROMPT = """\
You are a technical writer specializing in Python API documentation.
Your task is to rewrite raw API documentation into a compact, structured format \
that a code-generation model can use as a quick reference.

Rules:
- Be concise. Omit verbose prose, long warnings, deprecation notes, and See Also sections.
- Keep at most 1–2 short usage examples. Prefer the most representative one.
- Preserve exact parameter names, types, and default values.
- Preserve the return type and a one-line description of what it returns.
- Output only the rewritten doc — no preamble, no commentary.\
"""

_USER_TEMPLATE = """\
Rewrite the following API documentation into this exact format:

## `<full signature with defaults>`

**Description:** <one or two sentences — what it does and when to use it>

**Arguments:**
- `param` (`type`, default=`val`): <one-line description>
- ...  *(omit **kwargs / internal params)*

**Returns:** `type` — <one-line description>

**Example:**
```python
<minimal working example>
```

---

Raw documentation to rewrite:

{raw_doc}\
"""


class DocRewriter(Agent):
    """
    Rewrites raw API documentation into a compact, structured format
    suitable for use as RAG context in a code-generation prompt.

    One document in → one rewritten document out.

    Parameters
    ----------
    url : str
        Base URL of the OpenAI-compatible inference endpoint.
    temperature : float
        Sampling temperature. Low values (0.1–0.2) give stable, deterministic rewrites.
    max_tokens : int
        Hard cap on rewritten doc length. 512 is usually enough; bump to 768
        for APIs with many parameters (e.g. sklearn estimators).
    """

    def __init__(
        self,
        url: str,
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_tokens: int = 512,
    ):
        super().__init__("doc_rewriter")

        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = self.client.models.list().data[0].id

        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def run(self, raw_doc: Text) -> Text:
        """
        Rewrite a single raw API doc into a compact structured format.

        Parameters
        ----------
        raw_doc : str
            The raw documentation text (scraped docstring, HTML-to-text, etc.).

        Returns
        -------
        str
            Rewritten documentation in the structured markdown format.
        """
        user_prompt = _USER_TEMPLATE.format(raw_doc=raw_doc.strip())

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content.strip()
