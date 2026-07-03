from enum import Enum
from src.agent_constructor.agent import Agent
from openai import OpenAI
from src.agent_constructor.core import Text
from src.tracing import Tracer
from transformers import AutoTokenizer, PreTrainedTokenizerFast


DEFAULT_SYSTEM_PROMPT = """Write a short code following the given format and indentation. Place the executable code between <code> and </code> tags, without any other non-executable things."""


class SOLVER_API(Enum):
    COMPLETIONS = 'completions'
    CHAT_COMPLETIONS = 'chat'


class DS1000Solver(Agent):
    def __init__(
            self, 
            url: str, 
            api: str = SOLVER_API.COMPLETIONS.value,
            system_prompt: str = DEFAULT_SYSTEM_PROMPT,
            temperature=0.2, 
            top_p=0.95, 
            max_tokens=1024,
            max_context_lenght: int = 24000,
            stop_tokens=["</code>", "# SOLUTION END"]):
        
        super().__init__("ds_1000_solver")
        
        if not api in SOLVER_API:
            raise ValueError(f"API: '{api}' is not a valid API! Use one of: {[item.value for item in SOLVER_API]}") 
        
        self.client = OpenAI(
            base_url=url,
            api_key="vllm"
        )

        self.api = api
        self.system_prompt = system_prompt

        self.model_name = self.client.models.list().data[0].id

        self.temperature=temperature
        self.top_p=top_p
        self.max_tokens=max_tokens
        self.stop_tokens=stop_tokens
        self.max_context_lenght = max_context_lenght

        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self.model_name)


    def run(self, task: Text, context: Text) -> Text:
        if context:
            task_tokens = len(self.tokenizer.encode(task, add_special_tokens=False))
            budget = self.max_context_lenght - task_tokens

            # Truncate whole chunks rather than cutting mid-text
            context = self._truncate_context(context, budget)
            prompt = self._create_prompt(task, context)
        else:
            prompt = task

        Tracer.set("final_prompt", prompt)

        match self.api:
            case SOLVER_API.COMPLETIONS.value:
                return self._call_completions(prompt)
            case SOLVER_API.CHAT_COMPLETIONS.value:
                Tracer.set("system_prompt", self.system_prompt)
                return self._call_chat(self.system_prompt, prompt)
            case _:
                raise ValueError(f"invalid API: '{self.api}'")
    
    def _call_completions(self, prompt) -> Text:
        completions = self.client.completions.create(
            seed=41,
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=self.stop_tokens
        )

        return completions.choices[0].text

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
            stop=self.stop_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )

        return completions.choices[0].message.content

    def _truncate_context(self, context: Text, token_budget: int) -> Text:
        tokens = self.tokenizer.encode(context, add_special_tokens=False)
        if len(tokens) <= token_budget:
            return context
        return self.tokenizer.decode(tokens[:token_budget], skip_special_tokens=True)

    def _create_prompt(self, task: Text, context: Text) -> Text:
        if context.strip():
            return (
                f"{context}\n\n"
                f"# Task ({DEFAULT_SYSTEM_PROMPT})\n\n"
                f"{task}"
            )
        return task
