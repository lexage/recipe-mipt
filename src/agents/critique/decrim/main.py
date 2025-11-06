from abc import ABC, abstractmethod
from typing import List

class Agent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"


class Decrim(Agent):
    """DeCRIM agent implementing the Decompose, Critique, and Refine pipeline."""
    
    def __init__(
        self, 
        name: str = "Decrim", 
        max_iterations: int = 10
    ):
        super().__init__(name)
        self.max_iterations = max_iterations
    
    def llm(self, prompt: str, model_type: str = "initial") -> str:
        # TODO: Replace with actual LLM call
        return ""
    
    def _parse_constraints(self, constraints_response: str) -> List[str]:
        """Parse constraints from LLM response."""
        # Extract numbered constraints from the response
        lines = constraints_response.strip().split('\n')
        constraint_list = []
        
        for line in lines:
            line = line.strip()
            # Look for lines starting with numbers (1., 2., etc.)
            if line and line[0].isdigit():
                # Remove the number and any following punctuation
                # TODO: парсит, только если нумерация с точкой. 
                constraint = line.split('.', 1)[1].strip() if '.' in line else line
                constraint_list.append(constraint)
        
        return constraint_list if constraint_list else ["Follow all instructions carefully"]
    
    def _create_critique_prompt(self, instruction: str, constraints: List[str], response: str) -> str:
        """Build prompt for critique step."""
        constraints_text = "\n".join([f"{i+1}. {constraint}" for i, constraint in enumerate(constraints)])
        
        return f"""
        You are an assistant whose job is to help me perform tasks. 
        I will give you an instruction and an AI assistant response. 
        The instruction includes some constraints to be followed by AI assistant while generating response. 
        Your task is to check and let me know which of the constraints are satisfied by the AI assistant response. 
        Please state short reasons on whether constraint is satisfied in the response or not. 
        Also include final answer as "Constraint followed" or "Constraint not followed" for each constraint.
        
        Instruction: {instruction}
        
        Constraints:
        {constraints_text}
        
        Assistant Response: {response}
        
        Please analyze each constraint one by one and provide your critique:
        """
        
    def _all_constraints_satisfied(self, critique_result: str) -> bool:
        """Check if all constraints are satisfied based on critique result."""
        # TODO: Add real parsing
        unsatisfied_indicators = ["not followed", "not satisfied", "unsatisfied", "failed", "missing"]
        
        for indicator in unsatisfied_indicators:
            if indicator in critique_result.lower():
                return False
        return True
    
    def _extract_unsatisfied_constraints(
        self, 
        critique_result: str, 
        all_constraints: List[str]
    ) -> List[str]:
        """Extract list of unsatisfied constraints from critique response."""
        # TODO: Add real extraction
        unsatisfied = []
        lower_result = critique_result.lower()
        
        for i, constraint in enumerate(all_constraints):
            constraint_lower = constraint.lower()
            if any(indicator in lower_result for indicator in [f"constraint {i+1} not", f"{constraint_lower} not"]):
                unsatisfied.append(constraint)
        
        return unsatisfied if unsatisfied else all_constraints
    
    def _create_refine_prompt(
        self, 
        instruction: str, 
        previous_response: str, 
        unsatisfied_constraints: List[str]
    ) -> str:
        """Build prompt for refine step."""
        constraints_text = ", ".join([f'"{constraint}"' for constraint in unsatisfied_constraints])
        
        return f"""
        You are provided an instruction, an AI response to the instruction and a feedback about the response. 
        Please correct the AI response according to the feedback provided.
        
        Instruction: {instruction}
        
        AI response: {previous_response}
        
        Feedback: Response did not follow {len(unsatisfied_constraints)} constraint(s): {constraints_text}
        
        Corrected response:
        """
    
    def run(self, instruction: str) -> str:
        """
        Args:
            instruction: User instruction with multiple constraints
            
        Returns:
            Final refined response
        """
        initial_prompt = f"""
        You are an AI assistant. Please respond to the following user instruction.
        Make sure to follow all the provided constraints.
        
        Instruction: {instruction}
        
        Response:
        """
        initial_response = self.llm(initial_prompt, model_type="initial")
        current_response = initial_response
        
        decompose_prompt = f"""
        You are an assistant whose job is to help me perform tasks. 
        I will give you an instruction that implicitly contains constraints to be followed. 
        Your task is to list the constraints provided by the user in an enumerated list format.
        
        Original Instruction: {instruction}
        
        Provided Constraints:
        """
        constraints = self.llm(decompose_prompt, model_type="decompose")
        constraint_list = self._parse_constraints(constraints)
        
        for _ in range(self.max_iterations):
            critique_prompt = self._create_critique_prompt(instruction, constraint_list, current_response)
            critique_result = self.llm(critique_prompt, model_type="critic")
            
            if self._all_constraints_satisfied(critique_result):
                break
                
            unsatisfied_constraints = self._extract_unsatisfied_constraints(critique_result, constraint_list)
            
            refine_prompt = self._create_refine_prompt(instruction, current_response, unsatisfied_constraints)
            current_response = self.llm(refine_prompt, model_type="refine")
        
        return current_response