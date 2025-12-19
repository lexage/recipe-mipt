import re
from abc import ABC, abstractmethod
from src.agent_constructor.context_engine import ContextAssembler
from src.agent_constructor.db import IDB

class Agent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"


class ReActAgent(Agent):
    """ReAct agent for automatic programming tasks."""
    
    def __init__(
        self, 
        name: str = "ReActAgent", 
        instruction: str = None, 
        examples: list = None, 
        max_iterations: int = 10, 
        db: IDB = None, 
        context_assembler: ContextAssembler = None
    ):
        super().__init__(name)
        self.instruction = instruction or self._get_default_instruction()
        self.examples = examples or []
        self.max_iterations = max_iterations
        self.db = db
        self.context_assembler = context_assembler
        self.memory = []
        
    def _get_default_instruction(self):
        """Return default instruction for the ReAct agent."""
        return """You are an AI programming assistant that solves coding tasks using the ReAct (Reasoning-Action) framework.

You must generate responses in the following format:

Think: [Your step-by-step reasoning about what to do next to make progress on the task]
Action: [The specific action to take at this step]

Available actions include but are not limited to:
- retrieve_context: Retrieve relevant context or examples from the database
- Any other action you deem necessary to solve the programming task

After each action, you will receive an observation with the result. Use these observations to inform your next steps.

Work through the problem systematically, breaking it down into manageable steps."""
    
    def make_prompt(self, task: str) -> str:
        """Create the prompt for the LLM."""
        prompt = self.instruction + "\n\n"
        
        # Add few-shot examples
        if self.examples:
            prompt += "Examples:\n"
            for example in self.examples:
                prompt += example + "\n\n"
            
        prompt += f"Task: {task}\n\n"
        
        # Add history
        for item_type, content in self.memory:
            prompt += f"{item_type}: {content}\n"
        
        prompt += "\nPlease respond in the format:\nThink: [your reasoning]\nAction: [action_name]"
        
        return prompt
    
    def llm(self, prompt: str) -> str:
        # TODO: Replace with actual LLM call
        return ""
    
    def _extract_thought_and_action(self, response: str) -> tuple:
        """Extract thought and action from the model response."""
        thought = ""
        action = ""
        
        thought_match = re.search(r'Thought:\s*(.*?)(?=\nAction:|\n*$)', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        action_match = re.search(r"Action:.*?(?=\n|$)", response, re.DOTALL)
        if action_match:
            action_text = action_match.group(0).strip()
            action = action_text.replace("Action:", "").strip()
        
        return thought, action
    
    def retrieve_context(self, task: str, top_k: int = 10) -> str:
        """Retrieve context from the database."""
        if self.db and self.context_assembler:
            retrieved = self.db.query(task, top_k)
            context = self.context_assembler.assemble(task, retrieved)
            return context
        return "No database or context assembler available."
    
    def _make_action_prompt(self, task: str, action: str, thought: str, items_num: int = 6) -> str:
        """Create a prompt for arbitrary actions."""
        prompt = f"""Task: {task}
        
Current reasoning: {thought}

Action to perform: {action}

Based on the task and reasoning above, please perform the requested action. Provide a helpful response that moves us closer to solving the programming task. 

After your response, please indicate if the task is complete by adding:
Finish: True
or if more work is needed:
Finish: False"""
        
        # Add relevant history context
        if self.memory:
            prompt += "\n\nRelevant history:\n"
            for item_type, content in self.memory[-items_num:]:  # Last items_num items for context
                prompt += f"{item_type}: {content}\n"
        
        return prompt
    
    def _extract_finish_flag(self, response: str) -> bool:
        """Extract finish flag from the model response."""
        finish_match = re.search(r'Finish:\s*(True|False)', response, re.IGNORECASE)
        if finish_match:
            return finish_match.group(1).lower() == 'true'
        return False
    
    def _remove_finish_flag(self, response: str) -> str:
        """Remove finish flag from the response to get clean observation."""
        cleaned_response = re.sub(r'Finish:\s*(True|False)[\.,]?\s*', '', response, flags=re.IGNORECASE)
        cleaned_response = cleaned_response.strip()
        return cleaned_response
    
    def _execute_action(self, action: str, task: str, thought: str) -> tuple:
        """Execute the specified action and return observation and finish flag."""
        if action == "retrieve_context":
            observation = self.retrieve_context(task)
            return observation, False
        
        else:
            # For any other action, call LLM with a prompt based on task, action, and memory
            prompt = self._make_action_prompt(task, action, thought)
            response = self.llm(prompt)
            
            # Extract finish flag and clean observation
            is_finish = self._extract_finish_flag(response)
            observation = self._remove_finish_flag(response)
            
            return observation, is_finish
    
    def run(self, task: str) -> str:
        """Run the ReAct agent to solve the programming task."""
        self.memory = []
        
        for _ in range(self.max_iterations):
            prompt = self.make_prompt(task)
            response = self.llm(prompt)
            thought, action = self._extract_thought_and_action(response)
            
            if not thought or not action:
                thought = "Unable to parse response. Considering what to do next."
                action = "analyze_task"
            
            self.memory.append(("Thought", thought))
            self.memory.append(("Action", action))
            
            observation, is_finish = self._execute_action(action, task, thought)
            self.memory.append(("Observation", observation))
            
            if is_finish:
                return observation
        
        # If max iterations reached, return the last observation
        return self.memory[-1][1] if self.memory else "Maximum iterations reached without solution."