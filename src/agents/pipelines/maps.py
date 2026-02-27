import json

from typing import List, Tuple
from openai import OpenAI

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text


class AlignerMAPS(Agent):
    """Alignes the caption, context, and question to ensure the safe integration of these elements."""

    def __init__(self, url: str = None, model_name: str = None, name: str = "maps_aligner"):
        super().__init__(name)
        
        self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, task: Text) -> Text:

        system_prompt = (
"""You are a text alignment specialist conducting structured analysis through Socratic interrogation. Systematically examine text pairs using this framework:
1. [Content Deconstruction]
”What core entities/events are explicitly stated in each text? What measurable attributes (quantifiers, temporal markers, causal verbs) define their characteristics?”
2. [Consistency Audit]
”Where might these texts exhibit:
a) Logical incompatibility (contradictory assertions)
b) Contextual divergence (conflicting timelines/locations)
c) Semantic dissonance (differentiated connotation scales)
d) Omission patterns (mutually exclusive missing elements)”
3. [Contextual Fusion]
”What implicit connections could synthesize a unified background framework? Which combinatory elements (chronological anchors, spatial references, causal chains) create non-conflicting narrative coherence?”
4. [Relevance Filtering]
”Through lexical-semantic mapping, which aligned components directly correspond to the question’s:
1) Key inquiry points
2) Required evidence types
3) Implicit knowledge domains
4) Potential inference pathways?”"""
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ],
            temperature=0,
        )

        return response.choices[0].message.content


class ScholarMAPS(Agent):
    """Researches the professional knowledge required by problems and exploring various hypotheses"""

    def __init__(self, url: str = None, model_name: str = None, name: str = "maps_scholar"):
        super().__init__(name)

        self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, task: Text) -> Text:

        system_prompt = (
"""You are a scientific knowledge retrieval system conducting structured inquiry through Socratic questioning. Process input data with this analytical framework:
1. [Problem Decomposition]
”What conceptual components constitute the question’s core demand? What technical terminology (domain-specific lexemes), operational parameters (variables/constants), and procedural verbs (analyze/calculate/compare) require epistemological grounding?”
2. [Knowledge Mining] ”For each identified component:
a) What fundamental axioms/theorems/laws from established scientific literature could operationally define it?
b) What measurable properties (equations/units/experimental protocols) are textually implied as relevant?
c) What contextual constraints (temporal/spatial/conditional clauses) limit knowledge scope?”
3. [Relevance Validation]
”For each candidate knowledge unit:
Does the source text contain explicit lexical anchors (technical terms/formula symbols) justifying its inclusion?
What textual evidence (descriptive adjectives/quantifiers/causal conjunctions) indicates required depth of explanation?
Are there implicit conceptual dependencies (prerequisite theories/mathematical tools) necessitating parallel retrieval?”
4. [Taxonomic Organization] ”How should validated knowledge be structured to mirror:
1) Problem-solving workflow steps
2) Hierarchical concept dependencies
3) Cross-domain interface points
4) Uncertainty quantification needs?”
Operational Protocol: Restrict to textually evidenced knowledge Mark confidence levels using [TextExplicit/ContextImplied/ExternalRequired] tags
Output as: 1) Knowledge Inventory Table (Concept-Definition-SourceAnchor)
2) Dependency Graph (Nodes=Concepts, Edges=Relations)
3) Gap Analysis Report (ExternalKnowledgeRequirements)."""
        )


        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ],
            temperature=0,
        )

        return response.choices[0].message.content

 
class SolverMAPS(Agent):
    """Gatheres all necessary information and resolving MSPs by selecting the most appropriate experimental approach"""

    def __init__(self, url: str = None, model_name: str = None, name: str = "maps_solver"):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, task: Text) -> Text:

        system_prompt = (
"""You are a scientific problem-solving system operating through Socratic dialectics. Engage in this structured inquiry process:
1. [Problem Framing]
”What is the absolute irreducible core of the question? What technical terms require operational definitions? What grammatical structures (comparatives/conditionals/quantifiers) dictate the solution’s form?”
2. [Evidence Audit] ”For each data source (question stem/options/text):
a) What measurable quantities (numerical ranges/units) are explicitly stated?
b) What causal relationships (if A then B/implies/proportional to) are textually encoded?
c) What constraints (assumptions/limitations/boundary conditions) are lexically embedded?”
3. [Reasoning Pathway]
”Through counterfactual testing:
Which axioms/theorems would become relevant if parameter X varied ±10%?
What observable contradictions emerge when applying hypothesis Y to the given data?
How do option components restrict valid inference trajectories?”
4. [Solution Validation] ”Does the proposed resolution:
1) Maintain dimensional homogeneity across all equations?
2) Satisfy all explicit boundary conditions?
3) Preserve logical consistency with given information?
4) Align with canonical scientific representations?”
Operational Protocol:
Document each reasoning step with evidence anchors (e.g., “Stem-Line5: v=Δx/Δt”).
Flag unresolved assumptions with [UnvalidatedPremise] tags
Output JSON structured as:
{ { ”process”: { { ”Phase 1”: ”[Framing] Identified core demand as… (Evidence: Q-Line2)”, ”Phase 2”: ”[Audit] Quantified parameters… (ConflictResolved: OptionC vs Text\\S3)”, ”Phase 3”: ”[Pathway] Eliminated hypothesis 
α
 due to… (TheoremRef: Maxwell-Eq)”, ”Phase 4”: ”[Validation] Verified dimensional consistency in…”, }, ”final_answer”: ”final result”} }"""
        )
    
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ],
            temperature=0,
        )

        return response.choices[0].message.content


class CriticMAPS(Agent):
    """Provides feedback and continuous correction throughout the solving process"""

    def __init__(self, url: str = None, model_name: str = None, name: str = "maps_critic"):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, task: Text) -> Tuple[List[int], Text]:

        system_prompt = (
"""You are a Socratic assessment engine conducting dialectical evaluation through this protocol:
1. [Triadic Interrogation Framework]
For each evaluation dimension (caption/alignment/knowledge/solution):
Existential Challenge:
”What absolute evidence anchors (line numbers/data points/theorem references) validate this component’s existence?”
Consistency Prosecution:
”Does internal logic maintain isomorphism across:
a) Input premises → Processing steps
b) Methodological choices → Domain standards
c) Assertions → Supporting evidence?”
Boundary Stress Test:
”What parametric variation (±10%) would collapse this component’s validity? Which fragility indicators emerge first?” 2.
[Metric Operationalization]
Score each dimension (1-5) using:
5 = Withstands three counterfactual scenarios
4 = Requires ≤1 assumption validation
3 = Needs 2-3 evidence reinforcements
2 = Contains structural contradictions
1 = Fails basic existence verification
3. [Improvement Synthesis]
Generate Socratic feedback per dimension:
For caption: ”What geometric/spatial relations lack quantifiable descriptors?”
For alignment: ”Which logical connective lacks cross-text co-reference?”
For knowledge: ”Which concept dependency lacks literature anchoring?”
For solution: ”What inference leap lacks isomorphic mapping?”
[Example of Desired Output as JSON]
{
"score": {"alignment": 4, "knowledge": 3, "solution": 5},
"need_feedback": true,
"worst_step": "knowledge", 
"feedback": {
    "alignment": "All elements appear logically consistent, but linking the same dipole formalism across both scenarios could tighten cross-text references.",
    "knowledge": "The derivation references standard dipole field formulas but lacks explicit citations to anchor the theoretical steps.",
    "solution": "Stating the direction of the resulting field more explicitly (e.g., along p-hat) would reinforce the isomorphic mapping between the dipole and its field."
    }
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
