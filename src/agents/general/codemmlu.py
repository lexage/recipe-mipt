import re
from enum import Enum
from src.agent_constructor.agent import Agent
from openai import OpenAI
from src.agent_constructor.core import Text
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError

DEFAULT_SYSTEM_PROMPT = """"""

T = TypeVar("T", bound=BaseModel)
from pydantic import BaseModel, StringConstraints
from typing import Annotated

class AnswerFormat(BaseModel):
    # strip_whitespace=True удалит пробелы перед проверкой длины
    answer: Annotated[str, StringConstraints(min_length=1, max_length=1, strip_whitespace=True)]

class RagFormat(BaseModel):
    # strip_whitespace=True удалит пробелы перед проверкой длины
    answer: str

class SOLVER_API(Enum):
    COMPLETIONS = 'completions'
    CHAT_COMPLETIONS = 'chat'


class CodeMMLUSolver(Agent):
    def __init__(
            self,
            url: str,
            context_after_task: bool,
            model_name: str = None,
            api: str = SOLVER_API.COMPLETIONS.value,
            system_prompt: str = DEFAULT_SYSTEM_PROMPT,
            temperature=0.2, 
            top_p=0.95, 
            max_tokens=1024,
            max_context_lenght: int = 24000,
            stop_tokens=["</code>", "# SOLUTION END"]):
        
        super().__init__("ds_1000_solver")
        stop_tokens = []
        temperature=0.2
        
        valid_apis = {item.value for item in SOLVER_API}
        if api not in valid_apis:
            raise ValueError(f"API: '{api}' is not a valid API! Use one of: {list(valid_apis)}")
        # if not api in SOLVER_API:
        #     raise ValueError(f"AдPI: '{api}' is not a valid API! Use one of: {[item.value for item in SOLVER_API]}") 
        
        self.client = OpenAI(
            base_url=url,
            api_key="vllm"
        )

        self.api = api
        self.system_prompt = system_prompt

        # Honour the model from config; fall back to whatever the server serves.
        self.model_name = model_name or self.client.models.list().data[0].id

        self.temperature=temperature
        self.top_p=top_p
        self.max_tokens=max_tokens
        self.stop_tokens=stop_tokens
        self.max_context_lenght = max_context_lenght

        self.context_after_task = context_after_task

        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self.model_name)


    def run(self, task: Text, context: Text, response_format: Type[T]) -> Text:

        tokens = self.tokenizer.encode(context, add_special_tokens=False)

        if len(tokens) > self.max_context_lenght:
            context = self.tokenizer.decode(tokens[:self.max_context_lenght])

        prompt = self._create_prompt(task, context)

        match self.api:
            case SOLVER_API.COMPLETIONS.value:
                return self._call_completions(prompt, response_format)
            case SOLVER_API.CHAT_COMPLETIONS.value:
                return self._call_chat(self.system_prompt, prompt)
            case _:
                raise ValueError(f"invalid API: '{self.api}'")
    
    @staticmethod
    def _extract_letter(text: Text) -> Text:
        """Robustly reduce any model output to a single choice letter A-D.

        Guards against the model emitting an explanation instead of a bare
        letter (the cause of the 'format' failures). Prefers the LAST
        standalone A-D token, which is normally the final answer.
        """
        if text is None:
            return ""
        s = str(text).strip()
        matches = re.findall(r"\b([ABCD])\b", s)
        if matches:
            return matches[-1]
        matches = re.findall(r"([ABCD])", s)
        if matches:
            return matches[-1]
        return s[:1]

    def _call_completions(self, prompt, response_format: Type[T]) -> Text:
        # Structured output: the schema (AnswerFormat) constrains the model to a
        # single character. Falls back to a plain call + letter extraction if the
        # server does not support structured parsing.
        try:
            completions = self.client.beta.chat.completions.parse(
                seed=41,
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                response_format=response_format,  # Передаем класс Pydantic
            )
            msg = completions.choices[0].message
            parsed = getattr(msg, "parsed", None)
            if parsed is not None and getattr(parsed, "answer", None):
                return self._extract_letter(parsed.answer)
            return self._extract_letter(msg.content)
        except Exception:
            completions = self.client.chat.completions.create(
                seed=41,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            return self._extract_letter(completions.choices[0].message.content)

    def _call_chat(self, system_pormpt, user_prompt) -> Text:
        messages = [
            {"role": "system", "content": system_pormpt},
            {"role": "user", "content": user_prompt},
        ]

        completions = self.client.chat.completions.create(
            seed=42,
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=self.stop_tokens
        )

        return self._extract_letter(completions.choices[0].message.content)

    def _create_prompt(self, task: Text, context: Text) -> Text:

        if self.context_after_task:
            return f"""Solve {task}\nUsing info from documentation:\n{context}"""
        return f"""Using info from documentation:\n{context}\n\nSolve {task}"""
