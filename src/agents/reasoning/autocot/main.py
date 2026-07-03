import sys
from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Union, Optional
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from src.agent_constructor.agent import Agent


# PROMPT_GET_REACT_TRAJECTORY = """You are an expert AI agent that solves problems by interleaving reasoning and tool use. 
# Follow the ReAct format EXACTLY. Do not add extra text, explanations, or markdown.

# Available tools:
# - llm(query: str): Generates code, functions, or reasoning steps.
# - db_search(query: str): Retrieves concrete code snippets and usage examples from a codebase or documentation index. Outputs consist STRICTLY of code blocks, file paths, and minimal metadata (e.g., chunk IDs, line ranges). NEVER returns explanatory prose, tutorials, API descriptions, or conversational text.

# Trajectory format:
# Thought 1: [brief reasoning about next step]
# Action 1: [tool_name]
# Action Input 1: {"key": "value"}
# Observation 1: [realistic, concise tool output or simulated result]

# Thought 2: ...
# Action 2: ...
# Action Input 2: ...
# Observation 2: ...

# When the problem is fully solved, end with:
# Thought N: [confirmation that solution is ready]
# Action N: Finish
# Action Input N: {}

# Generate the COMPLETE trajectory in one response. Simulate observations realistically based on the tool's purpose.
# Problem: <problem>
# Reference code: <reference_code>
# Let's think step by step and use the tools when necessary.
# """

# PROMPT_GET_REACT_TRAJECTORY = """You are an expert AI agent that generates data for few-shot examples. Your task is to output a complete problem-solving trajectory strictly following the ReAct framework and formatting rules.

# The output MUST be wrapped into the few-shot template structure.

# Available tools:
# - llm(query: str): Generates code, functions, or reasoning steps.
# - db_search(query: str): Retrieves concrete code snippets and usage examples from a codebase or documentation index.

# Output format template layout:
# === FEW-SHOT EXAMPLE [Number] ===
# TASK:
# Problem:
# <problem>

# A:
# <code>
# <reference_code or environment setup>
# </code>
# result = ... # put solution in this variable
# BEGIN SOLUTION
# <code>
# Thought 1: [brief reasoning about next step]
# Action 1: [tool_name]
# Action Input 1: {'query': '...'}
# [Optional, include with 50% probability] Is_final 1: False

# Observation 1: [realistic, concise tool output or simulated result. If db_search was used, simulate a professional index retrieval with [CHUNK ... | doc=...] and code lines]

# Thought 2: ...
# ...
# Thought N: [confirmation that solution is ready]
# Action N: Finish (or finish)
# Action Input N: {}
# [Optional, if Is_final was used earlier] Is_final N: True

# CRITICAL RULES FOR INTERNAL FORMATTING:
# 1. Action Input MUST be a Python dictionary using single quotes for strings: {'query': 'your query here'}. Do NOT use pure JSON format.
# 2. Simulated observations for 'db_search' must look like actual search chunks, for example:
#    "query: your query
#    retrieved context: [CHUNK 12345 | doc=6789]
#    In [1]: ...
#    Out[1]: ..."
# 3. Maintain clean code formatting inside markdown blocks where appropriate.
# 4. Generate the COMPLETE trajectory block from "=== FEW-SHOT EXAMPLE..." to the last action in one response. Do not truncate.

# Problem: <problem>
# Reference code / Setup: <reference_code>
# Let's think step by step and generate the complete few-shot training block now.
# """

PROMPT_GET_REACT_TRAJECTORY = """You are an expert AI agent that generates data for training few-shot examples. Your task is to output a complete problem-solving trajectory strictly following the ReAct framework and formatting rules.

Available tools:
- llm(query: str): Generates code, functions, or reasoning steps.
- db_search(query: str): Retrieves concrete code snippets and usage examples from a codebase or documentation index.

CRITICAL RULES FOR OUTPUT FORMATTING:
1. Output ONLY the filled template layout. Do NOT include any introductory text, introductory markdown, thoughts, explanations, or concluding remarks. 
2. Start your response directly with "=== FEW-SHOT EXAMPLE" and end it immediately after the final action block.
3. Action Input MUST be a Python dictionary using single quotes for strings: {'query': 'your query here'}. Do NOT use pure JSON format.
4. Simulated observations for 'db_search' must look like actual search chunks, for example:
   "query: your query
   retrieved context: [CHUNK 12345 | doc=6789]
   In [1]: ...
   Out[1]: ..."
5. Maintain clean code formatting inside markdown blocks where appropriate.
6. Generate the COMPLETE trajectory block from "=== FEW-SHOT EXAMPLE..." to the last action in one response. Do not truncate.

Input Data:
Problem: <problem>
Reference code / Setup: <reference_code>

Output Template Layout to fill and return (Output NOTHING else):
=== FEW-SHOT EXAMPLE ===
TASK:
Problem: <Insert problem here>

A:
<code>
<Insert reference_code or environment setup here>
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>
Thought 1: [brief reasoning about next step]
Action 1: [tool_name]
Action Input 1: {'query': '...'}
[Optional, include with 50% probability] Is_final 1: False

Observation 1: [realistic, concise tool output or simulated result. If db_search was used, simulate a professional index retrieval with [CHUNK ... | doc=...] and code lines]

Thought 2: ...
...
Thought N: [confirmation that solution is ready]
Action N: Finish
Action Input N: {}
[Optional, if Is_final was used earlier] Is_final N: True
</code>
"""

PROMPT_GET_REWOO_TRAJECTORY = """You are an expert AI agent that generates data for training few-shot examples. Your task is to output a complete problem-solving plan strictly following the ReWOO (Reasoning Without Observation) framework and formatting rules.

Available tools:
- llm(query: str): Generates code, functions, or reasoning steps.
- db_search(query: str): Retrieves concrete code snippets and usage examples from a codebase or documentation index.

CRITICAL RULES FOR OUTPUT FORMATTING:
1. Output ONLY the filled template layout. Do NOT include any introductory text, introductory markdown blocks, conversational thoughts, explanations, or concluding remarks.
2. Start your response directly with "=== FEW-SHOT EXAMPLE" and end it immediately after the final closing code markdown block `</footer>` or `</code>`.
3. The "BEGIN SOLUTION" section MUST contain a valid JSON object wrapped inside a markdown code block. The JSON must define a list of execution steps with 'step_id', 'plan', 'tool', 'args' (containing the dictionary query), 'evidence_tag', and 'depends_on'.
4. Maintain clean code formatting inside markdown blocks.
5. Generate the COMPLETE block from "=== FEW-SHOT EXAMPLE..." to the last line in one single response. Do not truncate.

Input Data:
Example Number: <example_number>
Problem: <problem>
Reference code / Setup: <reference_code>

Output Template Layout to fill and return (Output NOTHING else):
=== FEW-SHOT EXAMPLE <example_number> ===
TASK:
Problem:
<Insert problem here>

A:
<code>
<Insert reference_code or environment setup here>
</code>
result = ... # put solution in this variable

THOUGHT:
<Insert concise, step-by-step internal monologue explaining the plan layout and why these specific steps/tools are chosen to solve the problem>

BEGIN SOLUTION
<code>
{
  "steps": [
    {
      "step_id": 1,
      "plan": "<Insert explicit planning sentence for the first step>",
      "tool": "<llm or db_search>",
      "args": {"query": "<Insert query string for the tool>"},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "<Insert explicit planning sentence for the second step>",
      "tool": "<llm or db_search>",
      "args": {"query": "<Insert query string for the tool>"},
      "evidence_tag": "#E2",
      "depends_on": []
    }
  ]
}
</code>
"""

PROMPT_GET_REACT_TRAJECTORY_CODEMMLU = """You are an expert AI agent that generates data for training AUTO-COT few-shot examples on a MULTIPLE-CHOICE code benchmark (CodeMMLU). Each task gives a Python code problem and four candidate options labelled A, B, C, D; EXACTLY ONE is the reference-correct answer. Your task is to output a COMPACT decision trajectory following the ReAct framework, ending on the reference-correct letter.

Available tools:
- llm(query: str): Generates reasoning steps or compares options.
- db_search(query: str): Retrieves the canonical reference implementation and concrete code snippets from a codebase or documentation index.

HOW TO DECIDE THE LETTER
- Reject any option that changes BEHAVIOUR (wrong comparison/operator/variable, off-by-one, wrong precedence, a dropped guard the spec needs, a wrong formula or algorithm). Keep the option whose result matches EVERY example exactly.
- If two or more surviving options are FUNCTIONALLY EQUIVALENT, keep the CANONICAL / minimal form (the simplest standard idiom), judging by FORM not position.
- The trajectory MUST settle on the reference-correct letter provided below.

CRITICAL RULES FOR OUTPUT FORMATTING:
1. Output ONLY the filled template layout. Do NOT include any introductory text, introductory markdown, explanations, or concluding remarks.
2. Start your response directly with "=== FEW-SHOT EXAMPLE" and end it immediately after the final "Answer:" line.
3. Keep it AUTO-COT compact: a concise one-paragraph "Problem:" paraphrase and a one-line compact description per "Solution A/B/C/D".
4. Emit a MULTI-STEP trajectory: interleave one or more Thought/Action/Observation cycles (use db_search to retrieve the canonical reference, or llm to compare options) BEFORE the final finish step. Vary the number of intermediate steps across examples (typically 1-3); do NOT always finish on the first step. Number every step sequentially (Thought 1/Action 1/..., Thought 2/Action 2/...).
5. Action Input MUST be a Python dictionary using single quotes for strings: {'query': '...'}; for 'finish' it is the empty dict {}. Do NOT use pure JSON format.
6. Simulated observations for 'db_search' must look like actual search chunks, for example:
   "query: your query
   retrieved context: [CHUNK 12345 | doc=6789]
   In [1]: ...
   Out[1]: ..."
7. The final step MUST be 'finish' with "Is_final N: True", and the final line MUST be "Answer: <letter>" equal to the reference-correct letter.
8. Generate the COMPLETE block from "=== FEW-SHOT EXAMPLE..." to the "Answer:" line in one response. Do not truncate.

Input Data:
Problem (with the four options A/B/C/D): <problem>
Reference-correct answer: <reference_code>

Output Template Layout to fill and return (Output NOTHING else):
=== FEW-SHOT EXAMPLE ===
TASK:
Problem: <Insert a concise paraphrase of the problem here>

Solution A: <one-line compact description of option A>
Solution B: <one-line compact description of option B>
Solution C: <one-line compact description of option C>
Solution D: <one-line compact description of option D>

Thought 1: [compact reasoning about what to verify next: trace examples/boundaries, start rejecting the behaviour-changing options]
Action 1: db_search
Action Input 1: {'query': '...'}
Is_final 1: False

Observation 1: [realistic simulated retrieval with [CHUNK ... | doc=...] and code lines]

Thought 2: ...
Action 2: ...
Action Input 2: {'query': '...'}
Is_final 2: False

Observation 2: ...

Thought N: [confirmation that the reference-correct option is settled; state the single letter]
Action N: finish
Action Input N: {}
Is_final N: True
Answer: <reference-correct letter A/B/C/D>
"""

PROMPT_GET_REWOO_TRAJECTORY_CODEMMLU = """You are an expert AI agent that generates data for training AUTO-COT few-shot examples on a MULTIPLE-CHOICE code benchmark (CodeMMLU). Each task gives a Python code problem and four candidate options labelled A, B, C, D; EXACTLY ONE is the reference-correct answer. Your task is to output a COMPACT decision plan following the ReWOO (Reasoning Without Observation) framework, ending on the reference-correct letter.

Available tools:
- llm(query: str): Generates reasoning steps or compares options.
- db_search(query: str): Retrieves the canonical reference implementation and concrete code snippets from a codebase or documentation index.

HOW TO DECIDE THE LETTER (drive the plan toward this)
- Reject any option that changes BEHAVIOUR (wrong comparison/operator/variable, off-by-one, wrong precedence, a dropped guard the spec needs, a wrong formula or algorithm). Keep the option whose result matches EVERY example exactly.
- If two or more surviving options are FUNCTIONALLY EQUIVALENT, keep the CANONICAL / minimal form (the simplest standard idiom), judging by FORM not position.
- The plan MUST support settling on the reference-correct letter provided below.

CRITICAL RULES FOR OUTPUT FORMATTING:
1. Output ONLY the filled template layout. Do NOT include any introductory text, introductory markdown blocks, explanations, or concluding remarks.
2. Start your response directly with "=== FEW-SHOT EXAMPLE" and end it immediately after the final "Answer:" line.
3. Keep it AUTO-COT compact: a concise one-paragraph "Problem:" paraphrase, a one-line compact description per "Solution A/B/C/D", a SINGLE inline "THOUGHT:" line, then the plan.
4. Emit a MULTI-STEP plan: the "steps" list must contain one or more steps, and you should usually produce 2-3 steps (e.g. a db_search retrieval whose evidence a later llm/db_search step depends on). Vary the number of steps across examples; do NOT always emit exactly one step. Number step_id sequentially and wire later steps to earlier evidence via 'depends_on' (e.g. ["#E1"]).
5. The plan MUST be RAW JSON (no "BEGIN PLAN"/"BEGIN SOLUTION" header, no <code> markdown wrapper): a JSON object with a "steps" list; each step has 'step_id', 'plan', 'tool', 'args' (a dict, e.g. {"query": "..."}), 'evidence_tag', and 'depends_on'.
6. The final line MUST be "Answer: <letter>" and MUST equal the reference-correct letter.
7. Generate the COMPLETE block from "=== FEW-SHOT EXAMPLE..." to the "Answer:" line in one single response. Do not truncate.

Input Data:
Example Number: <example_number>
Problem (with the four options A/B/C/D): <problem>
Reference-correct answer: <reference_code>

Output Template Layout to fill and return (Output NOTHING else):
=== FEW-SHOT EXAMPLE <example_number> ===
TASK:
Problem: <Insert a concise paraphrase of the problem here>

Solution A: <one-line compact description of option A>
Solution B: <one-line compact description of option B>
Solution C: <one-line compact description of option C>
Solution D: <one-line compact description of option D>

THOUGHT: <Insert one concise line: reject the behaviour-changing options, name the surviving canonical one, and outline what the plan steps retrieve/decide>
{
  "steps": [
    {
      "step_id": 1,
      "plan": "<Insert explicit planning sentence, e.g. retrieve the canonical reference for the described routine>",
      "tool": "<llm or db_search>",
      "args": {"query": "<Insert query string for the tool>"},
      "evidence_tag": "#E1",
      "depends_on": []
    },
    {
      "step_id": 2,
      "plan": "<Insert explicit planning sentence, e.g. given reference #E1 decide which option matches, rejecting the behaviour-changing variants>",
      "tool": "<llm or db_search>",
      "args": {"query": "<Insert query string for the tool>"},
      "evidence_tag": "#E2",
      "depends_on": ["#E1"]
    }
  ]
}
Answer: <reference-correct letter A/B/C/D>
"""

PROMPT_GET_REWOO_TRAJECTORY_CODEMMLU_SOLVER = """You are an expert AI agent that generates data for training SOLVER few-shot examples for a ReWOO (Reasoning Without Observation) pipeline on a MULTIPLE-CHOICE code benchmark (CodeMMLU). The SOLVER stage receives an ALREADY-COMPLETED plan — a list of Plan/Evidence pairs — and must output the FINAL answer as a single letter A, B, C, or D (EXACTLY ONE is reference-correct). Your task is to produce one complete SOLVER example: the task, the completed Plan/Evidence bullets, and the final structured RESPONSE, ending on the reference-correct letter.

The evidence would have been gathered with these tools:
- llm(query: str): Generates reasoning steps or compares options.
- db_search(query: str): Retrieves the canonical reference implementation and concrete code snippets from a codebase or documentation index.

HOW THE SOLVER DECIDES THE LETTER
- Reject any option that changes BEHAVIOUR (wrong comparison/operator/variable, off-by-one, wrong precedence, a dropped guard the spec needs, a wrong formula or algorithm). Keep the option whose result matches EVERY example exactly.
- If two or more surviving options are FUNCTIONALLY EQUIVALENT, keep the CANONICAL / minimal form (the simplest standard idiom), judging by FORM not position.
- Use the Evidence only when it gives a concrete, verifiable reason; ignore evidence about style, naming, or theoretical edge cases that do not change which option is reference-correct.
- The RESPONSE MUST settle on the reference-correct letter provided below.

CRITICAL RULES FOR OUTPUT FORMATTING:
1. Output ONLY the filled template layout. Do NOT include any introductory text, introductory markdown, explanations, or concluding remarks.
2. Start your response directly with "=== FEW-SHOT EXAMPLE" and end it immediately after the closing brace of the RESPONSE JSON.
3. Keep it AUTO-COT compact: a concise one-paragraph "Problem:" paraphrase and a one-line compact description per "Solution A/B/C/D".
4. Emit a MULTI-STEP completed plan: one or more "- Plan:"/"- Evidence:" bullet pairs (usually 2-3), mirroring what the planner would have retrieved (a canonical reference via db_search, then a decision comparing the options). Each Plan/Evidence uses single quotes and a realistic, concise evidence string. Vary the number of pairs across examples; do NOT always emit exactly one pair.
5. RESPONSE MUST be a JSON object with exactly two keys: "thought" (a concise reconciliation of the evidence) and "response" (EXACTLY ONE character — A, B, C, or D). The "response" value MUST equal the reference-correct letter.
6. Generate the COMPLETE block from "=== FEW-SHOT EXAMPLE..." to the RESPONSE JSON in one response. Do not truncate.

Input Data:
Example Number: <example_number>
Problem (with the four options A/B/C/D): <problem>
Reference-correct answer: <reference_code>

Output Template Layout to fill and return (Output NOTHING else):
=== FEW-SHOT EXAMPLE <example_number> ===
TASK:
Problem: <Insert a concise paraphrase of the problem here>

Solution A: <one-line compact description of option A>
Solution B: <one-line compact description of option B>
Solution C: <one-line compact description of option C>
Solution D: <one-line compact description of option D>

- Plan: '<Insert the first planning sentence, e.g. retrieve the canonical reference for the described routine>'
- Evidence: '<Insert the realistic tool result for that step, e.g. the canonical reference implementation>'
- Plan: '<Insert the second planning sentence, e.g. decide which option matches the reference, rejecting the behaviour-changing variants>'
- Evidence: '<Insert the realistic reasoning result that names the surviving canonical option>'

RESPONSE:
{
  "thought": "<Insert a concise reconciliation: why the behaviour-changing options are rejected and why the surviving canonical option is the answer>",
  "response": "<reference-correct letter A/B/C/D>"
}
"""

class AutoCoT(Agent):
    def __init__(
        self,
        problems: Union[str, List[Dict[str, str]]],
        name: str = "AutoCoT", 
        encoder_name: str = "all-MiniLM-L6-v2", # is used in original implementation
        model_name: str = "Qwen/Qwen1.5-32B-Chat-AWQ",
        openai_api_base_url="http://localhost:7215/v1",
    ):
        super().__init__(name)

        self.llm_model = ChatOpenAI(
            model=model_name,
            openai_api_base=openai_api_base_url,
            openai_api_key="fake-key",
            temperature=0.7,
        )
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

    # def llm(self, prompt: str) -> str:
    #     # TODO: Replace with actual LLM call
    #     return ""

    def llm(self, prompt: str) -> str:

        messages = [HumanMessage(content=prompt)]

        response = self.llm_model.invoke(messages)
        return response.content
    
    def generate_rationale_from_scratch(self, problem: str) -> str:
        """Generate reasoning chain from scratch for a problem."""
        reasoning_prompt = f"Problem:\n{problem}\n\nReasoning:\n{self.cot_trigger}"
        return self.llm(reasoning_prompt)
    
#     def generate_rationale_from_code(self, problem: str, reference_code: str) -> str:
#         """Generate reasoning chain based on existing reference code."""
#         rationale_prompt = f"""Problem:\n{problem}\n\nReference code:\n{reference_code}
        
# Explain the reasoning behind this code implementation:\n{self.cot_trigger}
# """
#         return self.llm(rationale_prompt)
    
    def generate_rationale_from_code(self, problem: str, reference_code: str) -> str:
        """Generate reasoning chain based on existing reference code."""

        # rationale_prompt = PROMPT_GET_REACT_TRAJECTORY.replace("<problem>", problem).replace("<reference_code>", reference_code)
        # rationale_prompt = PROMPT_GET_REACT_TRAJECTORY.format(problem=problem, reference_code=reference_code)
        rationale_prompt = PROMPT_GET_REWOO_TRAJECTORY.replace("<problem>", problem).replace("<reference_code>", reference_code)

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
            
            # demo_text += f"Problem: {question}\n"
            # demo_text += f"Reasoning: {rationale}\n"
            # demo_text += f"Answer: {code}\n\n"
            
            # demo_text += f"Problem: {question}\n"
            demo_text += f"{rationale}\n"
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
        # and the number of steps in the reasoning)l
        if not cluster:
            return None, None
        
        min_dist_idx = np.argmin(cluster_dists)
        demo_idx = cluster[min_dist_idx]
        problem_data = self.problems[demo_idx]

        problem_text = problem_data["question"]
        reference_code = problem_data.get("code")
        
        return demo_idx, problem_text, reference_code

    def construct_demos(self, num_demos: int = 3) -> str:
        """
        Construct demonstrations using Auto-CoT method.
        Handles both scenarios: with and without reference code.
        """
        questions = [item["question"] for item in self.problems]

        clusters, clustered_dists = self.cluster_problems(questions, num_clusters=num_demos)
        
        demos = []
        demos_idx = []
        for i, cluster in enumerate(clusters):
            if cluster:
                demo_idx, selected_problem, reference_code = self.select_demo_from_cluster(cluster, clustered_dists[i])
                if selected_problem:
                    demo = self.generate_rationale_and_code(selected_problem, reference_code)
                    demos.append(demo)
                    demos_idx.append(demo_idx)

        demo_text = self.create_demo_text(demos)
        return demos_idx, demo_text

    # def run(self, task: str, num_demos: int = 4) -> str:
    #     """
    #     Args:
    #         task: The programming problem to solve
    #         num_demos: Number of demonstrations to use
    #     """
    #     demo_text = self.construct_demos(num_demos)
    #     prompt = f"{demo_text}Problem:\n{task}\n\nReasoning:\n{self.cot_trigger}"
    #     reasoning = self.llm(prompt)
        
    #     answer_prompt = f"{prompt}\n{reasoning}\n{self.direct_answer_trigger}"
    #     code_solution = self.llm(answer_prompt)
       
    #     return code_solution

    def run(self, num_demos: int = 4) -> str:
        """
        Args:
            num_demos: Number of demonstrations to use
        """
        demos_idx, demo_text = self.construct_demos(num_demos)
        return demos_idx, demo_text


if __name__=="""__main__""":

    encoder_name = "all-MiniLM-L6-v2"
    model_name = "Qwen/Qwen2.5-32B-Instruct"
    data_path = "/workspace/proj/grant/recipe-mipt/data/ds1000_auto_cot.json" 
    auto_cot_agent = AutoCoT(problems=data_path,
                             name="AutoCoT", 
                             encoder_name=encoder_name, # is used in original implementation
                             model_name=model_name,
                             openai_api_base_url="http://172.18.0.1:7215/v1",
                            )

    demos_idx, demos = auto_cot_agent.run(num_demos=3)

    final_example = {
        "example": demos,
        "ids_example": demos_idx
    }


    # with open(f"/workspace/proj/grant/recipe-mipt/data/ds1000/auto_cot/{encoder_name}_example_react_trajectory.json", "w", encoding="utf-8") as f:
    #     json.dump(final_example, f, indent=4, ensure_ascii=False)

    with open(f"/workspace/proj/grant/recipe-mipt/data/ds1000/auto_cot/{encoder_name}_example_rewoo_trajectory.json", "w", encoding="utf-8") as f:
        json.dump(final_example, f, indent=4, ensure_ascii=False)

