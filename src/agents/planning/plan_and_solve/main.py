from src.agent_constructor.agent import Agent


class PlanAndSolveAgent(Agent):
    def __init__(self, name: str = "PlanAndSolveAgent"):
        super().__init__(name)
    
    def make_plan_prompt(self, task: str) -> str:
        return f"""Analyze the following programming problem:\n{task}

Provide a step-by-step development plan that includes:
1. Requirements analysis and key constraints
2. Solution architecture design
3. Data structures and algorithms to use
4. Implementation approach
5. Testing strategy"""

    def llm(self, prompt: str) -> str:
        # TODO: Replace with actual LLM call
        return ""
    
    def run(self, task: str) -> str:
        plan_prompt = self.make_plan_prompt(task)
        plan = self.llm(plan_prompt)
        return plan