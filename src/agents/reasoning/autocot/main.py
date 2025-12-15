import json
import numpy as np
from typing import List, Dict, Union, Optional
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from src.agents.agent_constructor.agent import Agent


class AutoCoT(Agent):    
    def __init__(
        self,
        problems: Union[str, List[Dict[str, str]]],
        name: str = "AutoCoT", 
        encoder_name: str = "all-MiniLM-L6-v2" # is used in original implementation
    ):
        super().__init__(name)
        self.cot_trigger = "Let's think step by step."
        self.direct_answer_trigger = "Therefore, the code is:"
        self.problems = self.load_problems(problems)
        self.encoder = SentenceTransformer(encoder_name)
        
    def load_problems(
        self, 
        problems: Union[str, List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        """
        Load programming problems from file or use provided examples. Dataset can have 2 formats:
        - Only questions: [{"question": "..."}]
        - Questions with reference code: [{"question": "...", "code": "..."}]
        """
        if isinstance(problems, str):
            with open(problems, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        elif isinstance(problems, list):
            return problems
        else:
            print(f"Warning: Invalid problems type.")
            return []
        
    def llm(self, prompt: str) -> str:
        # TODO: Replace with actual LLM call
        return ""
        
    def generate_rationale_from_scratch(self, problem: str) -> str:
        """Generate reasoning chain from scratch for a problem."""
        reasoning_prompt = f"Problem:\n{problem}\n\nReasoning:\n{self.cot_trigger}"
        return self.llm(reasoning_prompt)
    
    def generate_rationale_from_code(self, problem: str, reference_code: str) -> str:
        """Generate reasoning chain based on existing reference code."""
        rationale_prompt = f"""Problem:\n{problem}\n\nReference code:\n{reference_code}
        
Explain the reasoning behind this code implementation:\n{self.cot_trigger}
"""
        return self.llm(rationale_prompt)
    
    def generate_code_from_rationale(self, problem: str, rationale: str) -> str:
        """Generate code based on reasoning chain."""
        code_prompt = f"Problem:\n{problem}\n\nReasoning:\n{rationale}\n{self.direct_answer_trigger}"
        return self.llm(code_prompt)
    
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
    
    def create_demo_text(self, demos: List[Dict[str, str]]) -> str:
        """Create demonstration text for few-shot prompting."""
        demo_text = ""
        for demo in demos:
            question = demo["question"]
            rationale = demo["rationale"]
            code = demo["code"]
            
            demo_text += f"Problem: {question}\n"
            demo_text += f"Reasoning: {rationale}\n"
            demo_text += f"Answer: {code}\n\n"
        
        return demo_text
    
    def cluster_problems(self, problems: List[str], num_clusters: int = 4) -> List[List[int]]:
        """
        Cluster programming problems by similarity using sentence transformers and K-means.
        """
        if not problems:
            return [[] for _ in range(num_clusters)]
        
        embeddings = self.encoder.encode(problems)
        
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
            if cluster:
                selected_problem, reference_code = self.select_demo_from_cluster(cluster, clustered_dists[i])
                if selected_problem:
                    demo = self.generate_rationale_and_code(selected_problem, reference_code)
                    demos.append(demo)
        
        demo_text = self.create_demo_text(demos)
        return demo_text
    
    def run(self, task: str, num_demos: int = 4) -> str:
        """
        Args:
            task: The programming problem to solve
            num_demos: Number of demonstrations to use
        """
        demo_text = self.construct_demos(num_demos)
        prompt = f"{demo_text}Problem:\n{task}\n\nReasoning:\n{self.cot_trigger}"
        reasoning = self.llm(prompt)
        
        answer_prompt = f"{prompt}\n{reasoning}\n{self.direct_answer_trigger}"
        code_solution = self.llm(answer_prompt)
       
        return code_solution