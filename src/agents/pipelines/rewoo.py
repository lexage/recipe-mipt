from openai import OpenAI

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text


class PlannerREWOO(Agent):
    """
    ReWOO Planner: given an initial task, produces a textual blueprint
    """

    def __init__(
        self,
        url: str,
        model_name: str,
        maximum_steps: int = 5,
        name: str = "rewoo_planner_agent",
    ):
        super().__init__(name)
        self.maximum_steps = maximum_steps

        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name

    def run(self, task: Text, context: Text = "") -> Text:
        """
        Produce a textual plan for the given task (one or more lines).
        """

        system_prompt = (
            "You are the Planner module in a ReWOO-style pipeline.\n"
            "Given a complex task, you must break it down into a small number of "
            "ordered plans (high-level sub-tasks).\n\n"
            "Each plan should:\n"
            "- be self-contained and understandable on its own;\n"
            "- describe what information to retrieve or what to compute;\n"
            "- be something that another agent (Worker) can execute independently.\n\n"
            f"Output at most {self.maximum_steps} steps, one per line, in the form:\n"
            "Step 1: ...\nStep 2: ...\n..."
        )

        user_context = f"\n\nAdditional context:\n{context}" if context else ""
        user_prompt = f"Task:\n{task}{user_context}\n\nProduce the plan steps now."

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        content = response.choices[0].message.content

        lines = [line for line in content.splitlines() if line.strip()]
        if len(lines) > self.maximum_steps:
            lines = lines[: self.maximum_steps]
        return "\n".join(lines)


class WorkerREWOO(Agent):
    """
    ReWOO Worker: for each single plan step, gathers/produces supporting evidence.
    The pipeline can call this in parallel for different plan steps.
    """

    def __init__(
        self,
        url: str,
        model_name: str,
        name: str = "rewoo_worker_agent",
    ):
        super().__init__(name)

        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name

    async def run(self, task: Text, context: Text = "") -> Text:
        """
        Given the original task and a single plan step, return concise textual evidence.
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the Worker module in a ReWOO-style pipeline. "
                        "Given the current sub-task and optional context, you retrieve or infer "
                        "concise evidence that helps solve the overall task. "
                        "Return only the evidence text, without extra commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Sub-task:\n{task}\n\n"
                        f"Context (may include the original task, prior steps, etc.):\n{context}\n\n"
                        "Return the evidence for this sub-task."
                    ),
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content


class SolverREWOO(Agent):
    """
    ReWOO Solver: given the task, the full plan and all collected evidence,
    synthesize the final answer.
    """

    def __init__(
        self,
        url: str,
        model_name: str,
        name: str = "rewoo_solver_agent",
    ):
        super().__init__(name)

        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name

    def run(self, task: Text, context: Text = "") -> Text:
        """
        Combine the task and provided textual context (typically plans + evidences)
        into a final answer.
        """

        system_prompt = (
            "You are the Solver module in a ReWOO-style pipeline.\n"
            "You are given:\n"
            "- the original task,\n"
            "- and a block of context that already combines plans and evidences.\n\n"
            "Use this context carefully (it may contain irrelevant details) "
            "to answer the task as accurately as possible.\n"
            "Respond with the final answer only, without showing intermediate reasoning."
        )

        user_prompt = (
            f"Task:\n{task}\n\n"
            f"Context (plans + evidences):\n{context}\n\n"
            "Now provide the final answer to the task."
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        return response.choices[0].message.content
