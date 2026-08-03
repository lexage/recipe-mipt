from src.agent_constructor.agent import Agent
from openai import OpenAI
from src.agent_constructor.core import Text


_API_INSTRUCT_SYSTEM_PROMPT = """\
You evaluate Python API documentation for a programming task.

Rules:
- Do NOT write solution code, pseudocode, or a step-by-step solution to the task
- Do NOT answer or solve the task — only discuss the API's relevance and usage
- Base your assessment only on the provided documentation
- If the documentation is unrelated, say so briefly

Respond in this format:

**Usefulness:** high | medium | low | none
**How to use:** <1-3 sentences on how this API could help — typical calls, key parameters, patterns. No code for the full task.\
"""

_API_INSTRUCT_USER_TEMPLATE = """\
[PROGRAMMING TASK]
{task}

[API]
{api}

[API DOCUMENTATION]
{documentation}\
"""


class InstructRationalityAgent(Agent):
    def __init__(
            self, url: str, model_name: str, 
            temperature: float = 0.2,
            top_p: float = 0.95,
            max_tokens: float = 1024,
            stop_tokens: list = []
            ):
        
        super().__init__("instruct_rationality_agent")
        
        self.client = OpenAI(
            base_url=url,
            api_key="vllm"
        )

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop_tokens = stop_tokens


    def run(self, task: Text, context: Text) -> Text:

        prompt = f"""Read the following documents relevant to the given task:
[TASK] 
{task}

[DOCUMENT]
{context}

Please identify wether documnet is useful to answer the given task or no, and explain how the
contents lead to the answer or why it is non relevant.
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {'role': 'user', 'content': prompt}
            ]
            ,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=self.stop_tokens
        )

        return response.choices[0].message.content


class APIInstructRationalityAgent(Agent):
    """
    Produces a short rationale for one API doc chunk: usefulness for the task
    and how the API could be used, without solving the task.
    """

    def __init__(
        self,
        url: str,
        model_name: str | None = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 384,
        stop_tokens: list | None = None,
    ):
        super().__init__("api_instruct_rationality_agent")

        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name or self.client.models.list().data[0].id
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop_tokens = stop_tokens or []

    def run(self, task: Text, api: Text, documentation: Text) -> Text:
        user_prompt = _API_INSTRUCT_USER_TEMPLATE.format(
            task=task.strip(),
            api=api.strip(),
            documentation=documentation.strip(),
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": _API_INSTRUCT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=self.stop_tokens,
        )

        return response.choices[0].message.content.strip()
