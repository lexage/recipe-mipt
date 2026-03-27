import json
import logging

from typing import List, Tuple
from openai import OpenAI

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text


_LOG_SEPARATOR = f"\n{'_' * 20}\n"


class AlignerMAPS(Agent):
    """Alignes the caption, context, and question to ensure the safe integration of these elements."""

    def __init__(self, url: str = None, model_name: str = None, name: str = "maps_aligner"):
        super().__init__(name)
        
        self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, task: Text) -> Text:

#         system_prompt = (
# """You are a text alignment specialist conducting structured analysis through Socratic interrogation. Systematically examine text pairs using this framework:
# 1. [Content Deconstruction]
# ”What core entities/events are explicitly stated in each text? What measurable attributes (quantifiers, temporal markers, causal verbs) define their characteristics?”
# 2. [Consistency Audit]
# ”Where might these texts exhibit:
# a) Logical incompatibility (contradictory assertions)
# b) Contextual divergence (conflicting timelines/locations)
# c) Semantic dissonance (differentiated connotation scales)
# d) Omission patterns (mutually exclusive missing elements)”
# 3. [Contextual Fusion]
# ”What implicit connections could synthesize a unified background framework? Which combinatory elements (chronological anchors, spatial references, causal chains) create non-conflicting narrative coherence?”
# 4. [Relevance Filtering]
# ”Through lexical-semantic mapping, which aligned components directly correspond to the question’s:
# 1) Key inquiry points
# 2) Required evidence types
# 3) Implicit knowledge domains
# 4) Potential inference pathways?”"""
#         )

        system_prompt = (
        """You are a code alignment specialist conducting structured analysis through Socratic interrogation. Your goal: ensure precise alignment between problem specification, provided context/code scaffolding, and the required code output format.

        1. [Task Deconstruction]
        "What is the core computational objective? Identify:
        - Input contract: data types, structures, constraints, and preconditions explicitly stated or implied
        - Output contract: exact format required (e.g., variable assignment, function definition, code block delimiters), return type, side-effect expectations
        - Transformation logic: what algorithmic or structural operation maps input → output? (selection, transformation, aggregation, iteration, recursion, composition)"

        2. [Context Consistency Audit]
        "Check for mismatches between:
        a) Problem description ↔ starter code: Are all referenced identifiers defined? Are imports/dependencies complete or inferable?
        b) Starter code ↔ expected output: Does the placeholder pattern align with the solution structure (e.g., 'result = ...' → assignment expected)?
        c) Example I/O ↔ general case: Does the sample demonstrate the general pattern, or are there hidden edge cases not shown?"

        3. [Format Fusion]
        "What structural elements guarantee output compliance?
        - Mandatory prefixes/suffixes: exact strings that must appear (e.g., 'BEGIN SOLUTION', variable names, delimiters)
        - Code encapsulation: should logic be wrapped in a function/class for reusability, or is inline code expected?
        - Purity expectations: must the solution avoid mutating inputs? Should it handle copies or views?
        - Language idioms: which syntactic patterns are conventional for this operation in the target language?"

        4. [Relevance Filtering]
        "Map task components to implementation priorities:
        1) Core operation: the minimal code construct that achieves the transformation
        2) Safety considerations: input validation, error handling, edge case coverage (as required by task scope)
        3) Idiomatic expression: prefer language-native constructs over verbose workarounds
        4) Verifiability: ensure the solution can be tested against provided examples or derived test cases"

        Operational Protocol:
        - Flag ambiguities with [NeedsClarification] and specify what information is missing
        - Flag format risks with [FormatRisk] and suggest corrective structure
        - Output alignment report as JSON: {task_core, input_contract, output_contract, format_requirements, identified_risks}
        """
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ],
            temperature=0,
        )
        
        logging.info(f"ALIGNER SYSTEM PROMPT:\n\n{system_prompt}")
        logging.info(_LOG_SEPARATOR)
        logging.info(f"ALIGNER USER PROMPT:\n\n{task}")
        logging.info(_LOG_SEPARATOR)


        return response.choices[0].message.content


class ScholarMAPS(Agent):
    """Researches the professional knowledge required by problems and exploring various hypotheses"""

    def __init__(self, url: str = None, model_name: str = None, name: str = "maps_scholar"):
        super().__init__(name)

        self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, task: Text) -> Text:

#         system_prompt = (
# """You are a scientific knowledge retrieval system conducting structured inquiry through Socratic questioning. Process input data with this analytical framework:
# 1. [Problem Decomposition]
# ”What conceptual components constitute the question’s core demand? What technical terminology (domain-specific lexemes), operational parameters (variables/constants), and procedural verbs (analyze/calculate/compare) require epistemological grounding?”
# 2. [Knowledge Mining] ”For each identified component:
# a) What fundamental axioms/theorems/laws from established scientific literature could operationally define it?
# b) What measurable properties (equations/units/experimental protocols) are textually implied as relevant?
# c) What contextual constraints (temporal/spatial/conditional clauses) limit knowledge scope?”
# 3. [Relevance Validation]
# ”For each candidate knowledge unit:
# Does the source text contain explicit lexical anchors (technical terms/formula symbols) justifying its inclusion?
# What textual evidence (descriptive adjectives/quantifiers/causal conjunctions) indicates required depth of explanation?
# Are there implicit conceptual dependencies (prerequisite theories/mathematical tools) necessitating parallel retrieval?”
# 4. [Taxonomic Organization] ”How should validated knowledge be structured to mirror:
# 1) Problem-solving workflow steps
# 2) Hierarchical concept dependencies
# 3) Cross-domain interface points
# 4) Uncertainty quantification needs?”
# Operational Protocol: Restrict to textually evidenced knowledge Mark confidence levels using [TextExplicit/ContextImplied/ExternalRequired] tags
# Output as: 1) Knowledge Inventory Table (Concept-Definition-SourceAnchor)
# 2) Dependency Graph (Nodes=Concepts, Edges=Relations)
# 3) Gap Analysis Report (ExternalKnowledgeRequirements)."""
#         )

        system_prompt = (
        """You are a programming knowledge retrieval system using Socratic questioning to ground implementation decisions in established software engineering principles.

        1. [Problem Decomposition]
        "Break down the coding demand:
        - Technical concepts: what algorithmic pattern, data structure, or language feature is central to the solution?
        - Operational parameters: what are the types, shapes, and constraints of inputs/outputs?
        - Procedural semantics: what does 'solve this' mean operationally? (compute, transform, validate, iterate, compose)"

        2. [Knowledge Mining]
        "For each component, retrieve grounded knowledge:
        a) Core principle: what fundamental concept (indexing, mapping, filtering, recursion, composition) operationally defines the solution?
        b) Measurable properties: 
        - Computational complexity: time/space expectations implied by task scale
        - Determinism: should output be reproducible for identical inputs?
        - Mutability: does the operation modify state or return new values?
        c) Constraints: 
        - Language version or feature availability (if inferable from context)
        - Side-effect policies (pure function vs. stateful operation)
        - Format constraints (exact variable names, code delimiters, structural requirements)"

        3. [Relevance Validation]
        "Validate each knowledge unit against task evidence:
        - Lexical anchors: do specific terms in the problem ('shuffle', 'aggregate', 'validate') point to known patterns?
        - Example I/O: does the sample input/output confirm or refute candidate approaches?
        - Starter code: do existing imports, variable names, or structure narrow the solution space?
        - Implicit assumptions: what is taken for granted (valid inputs, single-threaded execution, memory availability)?"

        4. [Taxonomic Organization]
        "Structure knowledge for implementation:
        1) Workflow: sequence of operations from input parsing to output formatting
        2) Dependencies: language features, standard library components, or external modules required
        3) Interfaces: function signatures, class contracts, or module boundaries that enable testing and reuse
        4) Uncertainty handling: which assumptions are unverified and should be flagged or documented?"

        Operational Protocol:
        - Tag knowledge confidence: [TaskExplicit] / [ContextImplied] / [ExternalAssumption]
        - Avoid library-specific recommendations unless explicitly mentioned in task
        - Output as: 
        1) Knowledge Table: {concept, definition, evidence_anchor}
        2) Dependency List: [required_language_features, inferred_modules]
        3) Gap Report: [unverified_assumptions, optional_enhancements]
        """
        )


        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ],
            temperature=0,
        )

        logging.info(f"SCHOLAR SYSTEM PROMPT:\n\n{system_prompt}")
        logging.info(_LOG_SEPARATOR)
        logging.info(f"SCHOLAR USER PROMPT:\n\n{task}")
        logging.info(_LOG_SEPARATOR)
        
        return response.choices[0].message.content

 
class SolverMAPS(Agent):
    """Gatheres all necessary information and resolving MSPs by selecting the most appropriate experimental approach"""

    def __init__(self, url: str = None, model_name: str = None, name: str = "maps_solver"):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, task: Text) -> Text:

#         system_prompt = (
# """You are a scientific problem-solving system operating through Socratic dialectics. Engage in this structured inquiry process:
# 1. [Problem Framing]
# ”What is the absolute irreducible core of the question? What technical terms require operational definitions? What grammatical structures (comparatives/conditionals/quantifiers) dictate the solution’s form?”
# 2. [Evidence Audit] ”For each data source (question stem/options/text):
# a) What measurable quantities (numerical ranges/units) are explicitly stated?
# b) What causal relationships (if A then B/implies/proportional to) are textually encoded?
# c) What constraints (assumptions/limitations/boundary conditions) are lexically embedded?”
# 3. [Reasoning Pathway]
# ”Through counterfactual testing:
# Which axioms/theorems would become relevant if parameter X varied ±10%?
# What observable contradictions emerge when applying hypothesis Y to the given data?
# How do option components restrict valid inference trajectories?”
# 4. [Solution Validation] ”Does the proposed resolution:
# 1) Maintain dimensional homogeneity across all equations?
# 2) Satisfy all explicit boundary conditions?
# 3) Preserve logical consistency with given information?
# 4) Align with canonical scientific representations?”
# Operational Protocol:
# Document each reasoning step with evidence anchors (e.g., “Stem-Line5: v=Δx/Δt”).
# Flag unresolved assumptions with [UnvalidatedPremise] tags
# Output JSON structured as:
# { { ”process”: { { ”Phase 1”: ”[Framing] Identified core demand as… (Evidence: Q-Line2)”, ”Phase 2”: ”[Audit] Quantified parameters… (ConflictResolved: OptionC vs Text\\S3)”, ”Phase 3”: ”[Pathway] Eliminated hypothesis 
# α
#  due to… (TheoremRef: Maxwell-Eq)”, ”Phase 4”: ”[Validation] Verified dimensional consistency in…”, }, ”final_answer”: ”final result”} }"""
#         )

        system_prompt = (
        """You are a code generation system using Socratic dialectics to produce correct, formatted solutions.

        1. [Problem Framing]
        "Extract the irreducible core:
        - Goal: what single computational result must the code produce?
        - Key operation: what minimal language construct achieves this (indexing, comprehension, higher-order function, control flow)?
        - Format constraint: what exact textual pattern must the output follow (prefixes, delimiters, variable assignments)?
        - Purity requirement: must the solution avoid side effects, or is mutation acceptable per task scope?"

        2. [Evidence Audit]
        "Analyze provided materials:
        a) Example input/output: what transformation pattern does the sample demonstrate? Generalize to arbitrary valid inputs.
        b) Starter code: what identifiers, structures, or placeholders are provided? What must be completed vs. replaced?
        c) Reference patterns (if given): what structural template should the solution follow (function wrapper, inline expression, class method)?
        d) Constraints: what is explicitly excluded (no external libraries, no recursion, no mutation)?"

        3. [Reasoning Pathway]
        "Test alternatives counterfactually:
        - Option A: [candidate approach] → does it satisfy correctness, format, and constraints?
        - Option B: [alternative approach] → what edge cases or format requirements does it violate?
        - Format test: does the candidate output begin with required prefix and contain code in specified delimiters?
        - Minimality test: is this the simplest expression that meets all requirements?"

        4. [Solution Validation]
        "Verify proposed code against criteria:
        1) Correctness: does it produce the exact output for provided examples and generalized valid inputs?
        2) Format compliance: does output match required textual structure character-for-character where specified?
        3) Safety: does it respect mutability expectations and avoid unintended side effects?
        4) Idiomatic expression: does it use conventional language patterns rather than obscure workarounds?
        5) Verifiability: can the solution be tested against derived cases (empty input, boundary values, type variations)?"

        Operational Protocol:
        - Document reasoning with evidence anchors: "Chose X over Y because [task line N] specifies Z"
        - Flag assumptions with [AssumesValidInput] / [AssumesSingleThreaded] / etc.
        - Output JSON with EXACT code placement:
        {
        "process": {
            "Phase 1": "[Framing] Core goal and format constraints identified",
            "Phase 2": "[Audit] Example confirms pattern X; starter code provides Y",
            "Phase 3": "[Pathway] Rejected approach A due to [reason]; selected B",
            "Phase 4": "[Validation] Format matches; correctness verified on examples"
        },
        "final_answer": "Respond with the answer on task directly with no extra words."
        }
        """
        )
    
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ],
            temperature=0,
        )

        logging.info(f"SOLVER SYSTEM PROMPT:\n\n{system_prompt}")
        logging.info(_LOG_SEPARATOR)
        logging.info(f"SOLVER USER PROMPT:\n\n{task}")
        logging.info(_LOG_SEPARATOR)

        return response.choices[0].message.content


class CriticMAPS(Agent):
    """Provides feedback and continuous correction throughout the solving process"""

    def __init__(self, url: str = None, model_name: str = None, name: str = "maps_critic"):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, task: Text) -> Tuple[List[int], Text]:

#         system_prompt = (
# """You are a Socratic assessment engine conducting dialectical evaluation through this protocol:
# 1. [Triadic Interrogation Framework]
# For each evaluation dimension (caption/alignment/knowledge/solution):
# Existential Challenge:
# ”What absolute evidence anchors (line numbers/data points/theorem references) validate this component’s existence?”
# Consistency Prosecution:
# ”Does internal logic maintain isomorphism across:
# a) Input premises → Processing steps
# b) Methodological choices → Domain standards
# c) Assertions → Supporting evidence?”
# Boundary Stress Test:
# ”What parametric variation (±10%) would collapse this component’s validity? Which fragility indicators emerge first?” 2.
# [Metric Operationalization]
# Score each dimension (1-5) using:
# 5 = Withstands three counterfactual scenarios
# 4 = Requires ≤1 assumption validation
# 3 = Needs 2-3 evidence reinforcements
# 2 = Contains structural contradictions
# 1 = Fails basic existence verification
# 3. [Improvement Synthesis]
# Generate Socratic feedback per dimension:
# For caption: ”What geometric/spatial relations lack quantifiable descriptors?”
# For alignment: ”Which logical connective lacks cross-text co-reference?”
# For knowledge: ”Which concept dependency lacks literature anchoring?”
# For solution: ”What inference leap lacks isomorphic mapping?”
# [Example of Desired Output as JSON]
# {
# "score": {"alignment": 4, "knowledge": 3, "solution": 5},
# "need_feedback": true,
# "worst_step": "knowledge", 
# "feedback": {
#     "alignment": "All elements appear logically consistent, but linking the same dipole formalism across both scenarios could tighten cross-text references.",
#     "knowledge": "The derivation references standard dipole field formulas but lacks explicit citations to anchor the theoretical steps.",
#     "solution": "Stating the direction of the resulting field more explicitly (e.g., along p-hat) would reinforce the isomorphic mapping between the dipole and its field."
#     }
# }
# """
#         )

        system_prompt = (
        """You are a Socratic code assessment engine evaluating solutions through dialectical protocol.

        1. [Triadic Interrogation Framework]
        For each dimension (format_compliance/logical_correctness/code_quality):

        - Existential Challenge:
        "What evidence validates this component?
        • Format: Does output BEGIN with required prefix and contain code in specified delimiters?
        • Logic: Does the code produce exact output for provided examples? Can it be generalized?
        • Quality: Is code syntactically valid, readable, and consistent with language conventions?"

        - Consistency Prosecution:
        "Check isomorphism across:
        a) Problem statement → Implementation: do key terms map to appropriate constructs?
        b) Starter code → Solution: are placeholders replaced correctly without breaking structure?
        c) Example I/O → General case: does solution handle inputs beyond the sample?"

        - Boundary Stress Test:
        "What breaks the solution?
        • Empty/minimal inputs: does code handle gracefully or raise expected errors?
        • Boundary values: max/min sizes, null/None values, duplicate keys?
        • Format variations: does solution depend on exact string matching where flexibility is possible?
        • First fragility indicator: what minimal change to input would cause failure?"

        2. [Metric Operationalization]
        Score each dimension (1-5) using:
        5 = Perfect format + correct logic + idiomatic + handles 3+ edge cases + well-documented
        4 = Minor format tweak OR one unhandled non-critical edge case
        3 = Logic correct but format issues OR missing safety checks for stated constraints
        2 = Logical error (wrong algorithm/construct) OR major format violation
        1 = Syntax error / undefined identifier / completely wrong approach

        3. [Improvement Synthesis]
        Generate actionable Socratic feedback:
        - Format: "Output must begin with [exact string]; current starts with [actual] → restructure"
        - Logic: "Consider handling [edge case] if task scope permits; current assumes [premise]"
        - Quality: "[Construct] is correct but verbose; suggest [idiomatic alternative] for clarity"

        [Example Output JSON]
        {
        "score": {"format_compliance": 5, "logical_correctness": 5, "code_quality": 4},
        "need_feedback": true,
        "worst_step": "code_quality",
        "feedback": {
            "format_compliance": "Output correctly starts with required prefix and contains code in specified delimiters.",
            "logical_correctness": "Core transformation matches example I/O and generalizes to arbitrary valid inputs.",
            "code_quality": "Variable name 'x' lacks descriptiveness; consider 'result_indices' for clarity. Adding type annotations would improve maintainability."
        },
        "suggested_fixes": [
            "Rename ambiguous identifiers for clarity",
            "Add docstring or comment explaining non-obvious logic",
            "Optional: add input validation if used beyond task scope"
        ],
        "validation_tests": [
            "assert solution(input_example) == expected_output",
            "assert solution(empty_input) handles gracefully",
            "assert solution is deterministic for identical inputs"
        ]
        }
        """
        )

        user_prompt = (
            f"[Data to Evaluate]:\n{task}"
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content
        try:
            scores = self._parse_results(answer)
        except:
            scores = [-1, -1, -1]
            
        logging.info(f"CRITIC SYSTEM PROMPT:\n\n{system_prompt}")
        logging.info(_LOG_SEPARATOR)
        logging.info(f"CRITIC USER PROMPT:\n\n{user_prompt}")
        logging.info(_LOG_SEPARATOR)

        return scores, answer
    
    @staticmethod
    def _parse_results(answer: Text):
        data = json.loads(answer)
        score_dict = data["score"]
        return [
            score_dict["alignment"],
            score_dict["knowledge"],
            score_dict["solution"]
        ]