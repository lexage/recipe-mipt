from typing import List
from src.agent_constructor.agent import Agent


class LeastToMostPlanner(Agent):
    """
    Agent that implements Least-to-Most Prompting for automated programming tasks.
    Breaks down complex programming problems into simpler subproblems and solves them sequentially.
    """
    def __init__(self, name: str = "L2M_Planner"):
        super().__init__(name)
        self.subproblems = []
        
    def llm(self, prompt: str) -> str:
        # TODO: Replace with actual LLM call
        return ""
    
    def extract_subproblems(self, response: str) -> List[str]:
        # TODO: Add implementation
        return [""]
    
    def decompose(self, task: str) -> List[str]:
        """
        Decompose the main task into a sequence of simpler subproblems.
        Returns a list of subproblems.
        """
        decomposition_prompt = f"""
You are an expert at decomposing programming tasks. Break down the following programming task into a sequence of simpler subproblems that need to be solved in order.

Original task: {task}

Break down the task into subproblems such that:
1. Each subproblem is self-contained and solvable
2. The solution to each subproblem can be used in subsequent subproblems
3. The last subproblem should be the original task
4. Subproblems should progress from simple to complex

Response format - simply list the subproblems, each on a new line, starting with a number and period.

Subproblems:
"""
        
        response = self.llm(decomposition_prompt)
        subproblems = self.extract_subproblems(response)
        return subproblems
    
    def run(self, task: str) -> str:
        plan = self.decompose(task)
        return plan