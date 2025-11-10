from typing import List, Dict
from src.agents.agent_constructor.agent import Agent


class LeastToMostPlanner(Agent):
    """
    Agent that implements Least-to-Most Prompting for automated programming tasks.
    Breaks down complex programming problems into simpler subproblems and solves them sequentially.
    """
    
    def __init__(self, name: str = "L2M_Planner"):
        super().__init__(name)
        self.subproblems = []
        self.solutions = {}
    
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
    
    def solve(
        self, 
        subproblem: str, 
        previous_solutions: Dict[str, str], 
    ) -> str:
        """
        Solve a single subproblem using previous solutions as context.
        """
        context = ""
        if previous_solutions:
            context = "Previously solved subproblems and their solutions:\n"
            for i, (problem, solution) in enumerate(previous_solutions.items(), 1):
                context += f"{i}. Problem: {problem}\n   Solution: {solution}\n\n"
        
        solving_prompt = f"""
You are a programming expert. Solve the following subproblem using the context from previous solutions.

{context}
Current subproblem: {subproblem}

Important guidelines:
1. Be specific and provide working code or clear instructions
2. If it's code, provide complete, compilable functions
3. Consider solutions from previous subproblems
4. If the subproblem requires writing code, provide a ready-to-use implementation
5. If it's analysis or design, provide a clear plan or specification

Solution:
"""
        
        response = self.llm(solving_prompt)
        return response
    
    def run(self, task: str) -> str:
        # Step 1: Decomposition
        self.subproblems = self.decompose(task)
        
        # Step 2: Sequential solving
        self.solutions = {}
        solution = ""
        
        for i, subproblem in enumerate(self.subproblems, 1):
            solution = self.solve(subproblem, self.solutions)
            self.solutions[subproblem] = solution
        
        return solution