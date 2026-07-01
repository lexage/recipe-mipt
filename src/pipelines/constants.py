from enum import Enum


PIPELINES_IMPORT_PATH = "src.pipelines.templates"


class PipelinesNames(Enum):
    SIMPLE = "SimplePipeline"
    REWOO = "REWOOPipeline"
    MAPS = "MAPSPipeline"
    REACT = "REACTPipeline"
    PANEL = "PANELPipeline"
    MARS = "MARSPipeline"
    MARS_CRITIC_UPD = "MARSPipelineCriticUpd"


class ComponentNames(Enum):

    # ===== Agents: Critique =====
    COMPLEX_CRITIC_AGENT = "src.agents.critique.ComplexCritic"
    CRITIC_AGENT = "src.agents.critique.Critic"
    DECRIM_AGENT = "src.agents.critique.Decrim"
    REFLEXION_AGENT = "src.agents.critique.Reflexion"
    SELFREFINE_AGENT = "src.agents.critique.SelfRefine"

    # ===== Agents: General =====
    CODEMMLU_SOLVER_AGENT = "src.agents.general.CodeMMLUSolver"
    DS1000_SOLVER_AGENT = "src.agents.general.DS1000Solver"
    DUMMY_AGENT = "src.agents.general.DummyAgent"
    TFIDF_EMBEDDING = "src.agents.general.TFIDFEmbedding"
    EMBEDDING_AGENT = "src.agents.general.EmbeddingAgent"
    SIMPLE_AGENT = "src.agents.general.SimpleAgent"
    ICV_AGENT = "src.icl.icv.ICV"

    # ===== Agents: Pipelines =====
    REWOO_SOLVER = "src.agents.pipelines.SolverREWOO"
    REWOO_WORKER = "src.agents.pipelines.WorkerREWOO"
    REWOO_PLANNER = "src.agents.pipelines.PlannerREWOO"
    MAPS_SOLVER = "src.agents.pipelines.SolverMAPS"
    MAPS_SCHOLAR = "src.agents.pipelines.ScholarMAPS"
    MAPS_ALIGNER = "src.agents.pipelines.AlignerMAPS"
    MAPS_CRITIC = "src.agents.pipelines.CriticMAPS"
    REACT_AGENT = "src.agents.pipelines.ReActAgent"
    REACT_AGENT_SGR = "src.agents.pipelines.ReActAgentSGR"
    PANEL_AGENT = "src.agents.pipelines.PanelAgent"
    MARS_PLANNER = "src.agents.pipelines.PlannerMARS"
    MARS_TEACHER = "src.agents.pipelines.TeacherMARS"
    MARS_CRITIC = "src.agents.pipelines.CriticMARS"
    MARS_STUDENT = "src.agents.pipelines.StudentMARS"
    MARS_CRITIC_UPD = "src.agents.pipelines.CriticMARSUpd"


    # ===== Agents: Planning =====
    LEAST_TO_MOST_PLANNER = "src.agents.planning.LeastToMostPlanner"
    PLAN_AND_SOLVE_AGENT = "src.agents.planning.PlanAndSolveAgent"
    MPC_SAMPLE_AGENT = "src.agents.planning.MPCSampleAgent"

    # ===== Agents: RAG =====
    INSTRUCT_RATIONALITY_AGENT = "src.agents.rag.InstructRationalityAgent"
    API_INSTRUCT_RATIONALITY_AGENT = "src.agents.rag.APIInstructRationalityAgent"
    CORAG_FINAL_SOLVER = "src.agents.rag.CoRAGFinalSolver"
    API_CORAG_FINAL_SOLVER = "src.agents.rag.APICoRAGFinalSolver"
    RAPTOR_QA_AGENT = "src.agents.rag.RaptorQAAgent"
    RAPTOR_SUMM_AGENT = "src.agents.rag.RaptorSummarizationAgent"

    # ===== Agents: Generation =====
    API_SELECTOR = "src.agents.generation.APISelector"
    QUERY_GENERATOR = "src.agents.generation.QueryGenerator"

    # ===== Agents: Reasoning =====
    AUTO_COT_REASONING = "src.agents.reasoning.AutoCoT"
    CONTRASTIVE_COT_REASONING = "src.agents.reasoning.ContrastiveCoT"
    COT_REASONING = "src.agents.reasoning.CoT"
    SELECTION_INFERENCE_REASONING = "src.agents.reasoning.SelectionInference"
    
    # ===== DBs =====
    LOCAL_DB = "src.db.docs_db.LocalDB"
    LOCAL_RAPTOR_DB = "src.db.raptor_db.LocalRaptorDB"
    GITHUB_DB = "from src.db.github_db.GitHubDocsDB"

    # ===== Retrievers =====
    CORAG_RETRIEVER = "src.rag.corag.CoRAGRetriever"
    API_CORAG_RETRIEVER = "src.rag.corag.APICoRAGRetriever"
    RAPTOR_RETRIEVER = "src.rag.raptor.RaptorRetriever"
    INSTRUCT_RETRIEVER = "src.rag.instructrag.InstructRAGRetriever"
    API_INSTRUCT_RETRIEVER = "src.rag.instructrag.APIInstructRAGRetriever"
    SIMPLE_RETRIEVER = "src.rag.simple.SimpleRetriever"
    API_RETRIEVER = "src.rag.api.APIRetriever"

    # ===== Filters =====
    EXACT_SUBSTR_FILTER = "src.filtering.exactsubstr.ExactSubstrFiltrator"
    SIMPLE_LEXICAL_FILTER = "src.filtering.lexical_filtration.SimpleLexicalFiltrator"
    LENGTH_FILTER = "src.filtering.simple_filters.LengthFilter"
    EDUCATION_VALUE_FILTER = "src.filtering.textbooks_are_all_you_need.EducationValueClassifierFilter"

    # ===== Chunkers =====
    DUMMY_CHUNKER = "src.agent_constructor.chunkers.DummyChunker"
    SIMPLE_CHUNKER = "src.agent_constructor.chunkers.SimpleChunker"
    RECURSIVE_CHUNKER = "src.agent_constructor.chunkers.RecursiveChunker"

    # ===== Context Assemblers =====
    EXAMPLES_CONTEXT_ASSEMBLER = "src.context_assemblers.examples_assembler.ExamplesAssembler"
    CORAG_CONTEXT_ASSEMBLER = "src.context_assemblers.corag_assembler.CoRAGContextAssembler"
    SIMPLE_CONTEXT_ASSEMBLER = "src.agent_constructor.context_engine.SimpleContextAssembler"
    INSTRUCT_CONTEXT_ASSEMBLER = "src.context_assemblers.instruct_assembler.InstructRAGContextAssembler"

    # ===== ICL =====
    ICCL_ICL = "src.icl.iccl.ICCL"
    LENS_ICL = "src.icl.lens.Lens"
    FEW_SHOT_ICL = "src.icl.fewshot.FewShot"

    # ===== Generators =====
    CODE_EVAL_GENERATOR = "src.generation.code_eval.CodeEvalGenerator"
    INCORRECT_EXAMPLES_GENERATOR = "src.generation.incorrect_examples.IncorrectExampleGenerator"
    INSTRUCT_GENERATOR = "src.generation.new_insruct.InstructGenerator"
    RANDOM_WORD_GENERATOR = "src.generation.random_word.RandomWordGenerator"
    RANDOM_TOPIC_GENERATOR = "src.generation.random_topic.RandomTopicGenerator"
    ZERO_SHOT_GENERATOR = "src.generation.zero_shot.ZeroShotGenerator"
    ONE_SHOT_GENERATOR = "src.generation.one_shot.OneShotGenerator"

    # ===== TOOLS =====
    DB_SEARCH_TOOL = "src.tools.DBSearchTool"
    LLM_TOOL = "src.tools.LLMTool"
