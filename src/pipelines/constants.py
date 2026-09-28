from enum import Enum


PIPELINES_IMPORT_PATH = "src.pipelines.templates"


class PipelinesNames(Enum):
    SIMPLE = "SimplePipeline"
    SIMPLE_WITH_DOC_FILTER = "SimplePipelineWithDocFilter"
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
    CORAG_FINAL_SOLVER = "src.agents.rag.CoRAGFinalSolver"
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
    RAPTOR_RETRIEVER = "src.rag.raptor.RaptorRetriever"
    INSTRUCT_RETRIEVER = "src.rag.instructrag.InstructRAGRetriever"
    SIMPLE_RETRIEVER = "src.rag.simple.SimpleRetriever"
    API_RETRIEVER = "src.rag.api.APIRetriever"

    # ===== Filters =====
    EXACT_SUBSTR_FILTER = "src.filtering.exactsubstr.ExactSubstrFiltrator"
    SIMPLE_LEXICAL_FILTER = "src.filtering.lexical_filtration.SimpleLexicalFiltrator"
    LENGTH_FILTER = "src.filtering.simple_filters.LengthFilter"
    EDUCATION_VALUE_FILTER = "src.filtering.textbooks_are_all_you_need.EducationValueClassifierFilter"
    CODE_DENSITY_FILTER = "src.filtering.code_density.CodeDensityFilter"
    SEMANTIC_DEDUP_FILTER = "src.filtering.semantic_dedup.SemanticDedupFilter"
    AGGRESSIVE_PATTERN_FILTER = "src.filtering.aggressive_pattern.AggressivePatternFilter"
    PERPLEXITY_FILTER = "src.filtering.perplexity.PerplexityFilter"
    AST_COMPLEXITY_FILTER = "src.filtering.ast_complexity.ASTComplexityFilter"
    TFIDF_FILTER = "src.filtering.tfidf.TfIdfInformativenessFilter"

    # ===== Filters: experiment 3 — our new universal/code filters =====
    RANK_FUSION_FILTER = "src.filtering.rank_fusion.RankFusionFilter"
    COMPRESSION_FILTER = "src.filtering.compression.CompressionDensityFilter"
    COHERENCE_FILTER = "src.filtering.coherence.CoherenceFilter"
    BOILERPLATE_FILTER = "src.filtering.boilerplate.BoilerplateFilter"
    API_DOC_FILTER = "src.filtering.api_doc_density.ApiDocDensityFilter"

    # ===== Filters: experiment 4 — "document is its own reference" cleaners =====
    SELF_CLEAN_FILTER = "src.filtering.self_clean.SelfConsistencyCleaner"

    # ===== Filters: experiment 5 — realistic dirt; text/code-aware; topic-aware =====
    SELF_CLEAN_V2_FILTER = "src.filtering.self_clean_v2.SelfConsistencyCleanerV2"
    CODE_SELECT_FILTER = "src.filtering.code_aware.CodeAwareSelectCleaner"
    CODE_LLM_FILTER = "src.filtering.code_aware.CodeAwareLLMCleaner"
    TOPIC_SELECT_FILTER = "src.filtering.topic_aware.TopicSelectCleaner"
    TOPIC_LLM_FILTER = "src.filtering.topic_aware.TopicLLMCleaner"

    # ===== Document Filters (pre-chunking; only used by SIMPLE_WITH_DOC_FILTER) =====
    DOC_DEDUP_FILTER = "src.filtering.doc_dedup.DocumentDedupFilter"
    DOC_CODE_DENSITY_FILTER = "src.filtering.doc_code_density.DocumentCodeDensityFilter"
    DOC_SEMANTIC_DEDUP_FILTER = "src.filtering.doc_semantic_dedup.DocumentSemanticDedupFilter"
    DOC_PERPLEXITY_FILTER = "src.filtering.doc_perplexity.DocumentPerplexityFilter"

    # ===== Document Filters: experiment 6 — API-reference vs narrative genre =====
    API_SECTION_FILTER = "src.filtering.api_section.ApiReferenceSectionFilter"
    API_GENRE_FILTER = "src.filtering.api_genre.ApiReferenceGenreFilter"

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

    # ===== Generators: experiment 6 — oracle (paraphrase reference answers) =====
    PARAPHRASE_GENERATOR = "src.generation.paraphrase.ParaphraseGenerator"

    # ===== Generators: experiment 8 — code2doc (oracle) + code2task (corpus) =====
    CODE_TO_DOC_GENERATOR = "src.generation.code2doc.CodeToDocGenerator"
    CODE_TO_TASK_GENERATOR = "src.generation.code2task.CodeToTaskGenerator"

    # ===== Generators: experiment 13 — RAG guide generation (deployable) =====
    RAG_GUIDE_GENERATOR = "src.generation.rag_guides.RagGuideGenerator"

    # ===== Generators: experiment 15 — API-anchored instructions =====
    API_GUIDE_GENERATOR = "src.generation.api_guides.ApiGuideGenerator"

    # ===== Generators: intent-anchored corpus augmentation =====
    INTENT_BASED_GENERATOR = "src.generation.intent_based.IntentBasedGenerator"

    # ===== Generators: experiment 17 — methods derived from the exp16 ablation
    # Corpus-grounded how-to guides. context_docs=0 -> method 1 (parametric),
    # context_docs>0 -> method 2 (grounded in retrieved corpus passages).
    HOWTO_GUIDE_GENERATOR = "src.generation.howto_guides.HowtoGuideGenerator"
    # Method 4: writes from observed DS-1000 failures. Capability probe only —
    # its scores are tuned on the tasks it was shown.
    ORACLE_ERROR_GENERATOR = "src.generation.oracle_errors.OracleErrorGenerator"

    # ===== Writers: text generated INTO the solver's system prompt =====
    # Not a Generator: returns no documents. The LLM writes the rule text from
    # the problem descriptions the module carries.
    PROMPT_RULE_GENERATOR = "src.generation.prompt_rules.PromptRuleGenerator"

    # ===== TOOLS =====
    DB_SEARCH_TOOL = "src.tools.DBSearchTool"
    LLM_TOOL = "src.tools.LLMTool"
