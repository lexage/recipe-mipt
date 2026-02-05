from typing import List

from openai import OpenAI

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text


class PlannerREWOO(Agent):
    """
    ReWOO Planner: given an initial task, produces a short, ordered list
    of high-level plans (sub-tasks) that a Worker can execute independently.
    """

    def __init__(
        self,
        url: str | None = None,
        model_name: str | None = None,
        maximum_steps: int = 5,
        name: str = "rewoo_planner_agent",
    ):
        super().__init__(name)
        self.maximum_steps = maximum_steps

        # If no model/url provided, fall back to a simple deterministic stub.
        self.dummy_mode = not (url and model_name)
        if not self.dummy_mode:
            self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name

    def run(self, task: Text) -> List[Text]:
        """
        Produce an ordered list of textual plan steps for the given task.
        """
        if self.dummy_mode:
            # Simple baseline: repeat the original task with step numbers.
            return [f"Step {i + 1}: {task}" for i in range(self.maximum_steps)]

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

        user_prompt = f"Task:\n{task}\n\nProduce the plan steps now."

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        content = response.choices[0].message.content or ""
        lines = [line.strip() for line in content.splitlines() if line.strip()]

        # Truncate to the configured maximum number of steps.
        return lines[: self.maximum_steps]


class WorkerREWOO(Agent):
    """
    ReWOO Worker: for each single plan step, gathers/produces supporting evidence.
    The pipeline can call this in parallel for different plan steps.
    """

    def __init__(
        self,
        url: str | None = None,
        model_name: str | None = None,
        name: str = "rewoo_worker_agent",
    ):
        super().__init__(name)

        self.dummy_mode = not (url and model_name)
        if not self.dummy_mode:
            self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name

    async def run(self, task: Text, plan_step: Text) -> Text:
        """
        Given the original task and a single plan step, return concise textual evidence.
        """
        if self.dummy_mode:
            return f"Evidence for plan '{plan_step}' on task '{task}'"

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the Worker module in a ReWOO-style pipeline. "
                        "Given the original task and one plan step, you retrieve or infer "
                        "concise evidence that helps solve the task. "
                        "Return only the evidence text, without extra commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Task:\n{task}\n\n"
                        f"Plan step:\n{plan_step}\n\n"
                        "Return the evidence for this plan step."
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
        url: str | None = None,
        model_name: str | None = None,
        name: str = "rewoo_solver_agent",
    ):
        super().__init__(name)

        self.dummy_mode = not (url and model_name)
        if not self.dummy_mode:
            self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name

    def run(self, task: Text, plan: List[Text], evidences: List[Text]) -> Text:
        """
        Combine the task, plans and evidences into a final answer.
        """
        if self.dummy_mode:
            lines = []
            for i, (p, e) in enumerate(zip(plan, evidences), start=1):
                lines.append(f"{i}. Plan: {p}\n   Evidence: {e}")
            return "FINAL ANSWER (dummy solver):\n" + "\n\n".join(lines)

        plans_block = "\n".join(
            f"Plan {i + 1}: {p}\nEvidence {i + 1}: {e}"
            for i, (p, e) in enumerate(zip(plan, evidences))
        )

        system_prompt = (
            "You are the Solver module in a ReWOO-style pipeline.\n"
            "You are given:\n"
            "- the original task,\n"
            "- an ordered list of plan steps,\n"
            "- and evidence corresponding to each step.\n\n"
            "Use the plans and evidence carefully (they may contain irrelevant details) "
            "to answer the task as accurately as possible.\n"
            "Respond with the final answer only, without showing intermediate reasoning."
        )

        user_prompt = (
            f"Task:\n{task}\n\n"
            f"Plans and evidence:\n{plans_block}\n\n"
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
