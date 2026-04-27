import os
import sys
sys.path.insert(0, os.getcwd())

import json
import numpy as np
import re
from openai import OpenAI
from typing import List, Dict, Union, Optional
from sklearn.cluster import KMeans
from src.agent_constructor.agent import Agent


class AutoCoT(Agent):    
    def __init__(
        self,
        problems: Union[str, List[Dict[str, str]]],
        name: str = "AutoCoT", 
        model_url: str = "http://localhost:7215/v1",
        model_name: str = "Qwen/Qwen1.5-32B-Chat-AWQ",
        embed_url: str = "http://localhost:7216/v1",
        embed_name: str = "Qwen/Qwen3-Embedding-4B",
        temperature: float = 0
    ):
        super().__init__(name)
        self.cot_trigger = "Let's think step by step."
        self.direct_answer_trigger = "Therefore, the code is:"
        self.problems = self.load_problems(problems)
        self.model_name = model_name
        self.embed_name = embed_name
        self.temperature = temperature
        
        self.model_client = OpenAI(
            base_url=model_url,
            api_key="vllm"
        )
        
        self.embed_client = OpenAI(
            base_url=embed_url,
            api_key="vllm"
        )
        
    def load_problems(
        self, 
        problems: Union[str, List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        """
        Load programming problems from file or use provided examples. Dataset can have 2 formats:
        - Only questions: [{"question": "..."}]
        - Questions with reference code: [{"question": "...", "code": "..."}]
        """
        if isinstance(problems, str) and problems.endswith('.jsonl'):
            examples_list = []
            with open(problems, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        examples_list.append(json.loads(line))
            return examples_list
        elif isinstance(problems, list):
            return problems
        else:
            print(f"Warning: Invalid problems type.")
            return []
        
    def generate(self, user_prompt: str, system_prompt: str = "You are a helpful assistant") -> str:
        response = self.model_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
        )
        
        return response.choices[0].message.content
        
    def generate_rationale_from_scratch(self, problem: str) -> str:
        """Generate reasoning chain from scratch for a problem."""
        reasoning_prompt = f"Problem:\n{problem}\n\nReasoning:\n{self.cot_trigger}"
        return self.generate(reasoning_prompt)
    
    def generate_rationale_from_code(self, problem: str, reference_code: str) -> str:
        """Generate reasoning chain based on existing reference code."""
        rationale_prompt = f"""Problem:
{problem}

Solution code:
{reference_code}

Explain the reasoning behind this solution step by step. Imagine you are solving the problem yourself and describe your thought process. Do not refer to the solution as given; focus on the logical steps and why they are taken."""

        system_prompt = "You are an AI assistant that explains solutions to programming problems. When given a problem and a correct solution, you explain the reasoning behind the solution in a natural, step-by-step manner, as if you were the one solving it. Do not mention that the solution is provided; simply describe the thought process and the key steps that lead to the implementation."
        
        return self.generate(rationale_prompt, system_prompt=system_prompt)
    
    def generate_code_from_rationale(self, problem: str, rationale: str) -> str:
        """Generate code based on reasoning chain."""
        code_prompt = f"Problem:\n{problem}\n\nReasoning:\n{rationale}\n{self.direct_answer_trigger}"
        return self.generate(code_prompt)
    
    def generate_rationale_and_code(self, problem: str, reference_code: Optional[str] = None) -> Dict[str, str]:
        """
        Generate both reasoning and code for a given problem using LLM.
        
        Two scenarios:
        1. No reference code: Generate reasoning first, then code
        2. With reference code: Generate reasoning based on existing code
        """
        if reference_code:
            rationale = self.generate_rationale_from_code(problem, reference_code)
            code = reference_code
        else:
            rationale = self.generate_rationale_from_scratch(problem)
            code = self.generate_code_from_rationale(problem, rationale)
        
        return {
            "question": problem,
            "rationale": rationale,
            "code": code
        }
        
    def remove_problem_prefix(self, text: str) -> str:
        return re.sub(r'^Problem:\s*', '', text, flags=re.IGNORECASE)
    
    def create_demo_text(self, demos: List[Dict[str, str]]) -> str:
        """Create demonstration text for few-shot prompting."""
        demo_text = ""
        for i, demo in enumerate(demos):
            question = demo["question"]
            rationale = demo["rationale"]
            code = demo["code"]
            
            clean_question = self.remove_problem_prefix(question)
            
            demo_text += f"Example {i + 1}:\n\n"
            demo_text += f"Problem:\n{clean_question}\n\n"
            demo_text += f"Reasoning:\n{rationale}\n\n"
            demo_text += f"Solution:\n{code}\n\n"
        
        return demo_text
    
    def get_embeddings(self, problems: List[str]):
        response = self.embed_client.embeddings.create(
            model=self.embed_name,
            input=problems
        )
        
        embeddings = [item.embedding for item in response.data]
        return embeddings
    
    def cluster_problems(self, problems: List[str], num_clusters: int = 4) -> List[List[int]]:
        """
        Cluster programming problems by similarity using sentence transformers and K-means.
        """
        embeddings = self.get_embeddings(problems)
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        distances = kmeans.transform(embeddings)
        
        clusters = [[] for _ in range(num_clusters)]
        clustered_dists = [[] for _ in range(num_clusters)]
        
        for idx, cluster_id in enumerate(cluster_labels):
            clusters[cluster_id].append(idx)
            clustered_dists[cluster_id].append(distances[idx][cluster_id])
        
        return clusters, clustered_dists
    
    def select_demo_from_cluster(
        self, 
        cluster: List[int], 
        cluster_dists: List[float]
    ) -> tuple[str, Optional[str]]:
        """Select one representative problem from each cluster (the closest to the cluster centroid)."""
        # TODO: Add more complex criteria for selecting a demonstration 
        # (for example, based on the length of the question 
        # and the number of steps in the reasoning)
        if not cluster:
            return None, None
        
        min_dist_idx = np.argmin(cluster_dists)
        demo_idx = cluster[min_dist_idx]
        problem_data = self.problems[demo_idx]
        
        problem_text = problem_data["question"]
        reference_code = problem_data.get("code")
        
        return problem_text, reference_code
    
    def construct_demos(self, num_demos: int = 4) -> str:
        """
        Construct demonstrations using Auto-CoT method.
        Handles both scenarios: with and without reference code.
        """
        questions = [item["question"] for item in self.problems]

        clusters, clustered_dists = self.cluster_problems(questions, num_clusters=num_demos)
        
        demos = []
        for i, cluster in enumerate(clusters):
            selected_problem, reference_code = self.select_demo_from_cluster(cluster, clustered_dists[i])
            demo = self.generate_rationale_and_code(selected_problem, reference_code)
            demos.append(demo)
        
        demo_text = self.create_demo_text(demos)
        return demo_text
    
    def extract_solution(self, response: str) -> str:
        marker = "Solution:\n"
        idx = response.find(marker)
        if idx == -1:
            return "Solution not found"

        return response[idx + len(marker):]
    
    def run(self, task: str, num_demos: int = 4) -> str:
        """
        Args:
            task: The programming problem to solve
            num_demos: Number of demonstrations to use
        """
        demo_text = self.construct_demos(num_demos) 
        
        clean_task = self.remove_problem_prefix(task)
        
        prompt = f""""Here are some examples of solving programming problems step by step:
        
{demo_text}
Now analyze the following problem. Provide step-by-step reasoning. After your reasoning, write the solution code after the line 'Solution:'.

Problem:
{clean_task}

Reasoning:
"""
        
        system_prompt = (
            "You are a helpful programming assistant. When given a problem, you should first think step by step "
            "and then provide the solution code after the line 'Solution:'. "
            "Make sure to include exactly 'Solution:' on its own line followed by the code. "
            "Do not add any extra text after the code."
        )
        
        response = self.generate(prompt, system_prompt=system_prompt)
        solution = self.extract_solution(response)
        
        return solution