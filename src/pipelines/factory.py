from typing import Dict, Any

from src.pipelines.configs import AgentConfig, DBConfig, FilterConfig, RetrieverConfig, ChunkerConfig, ContextAssemblerConfig, AnyConfig

from src.agents.tfidf_agent import TFIDFEmbedding
from src.agents.dummy_agents import DummyPlanner, DummyQAAgent, DummySummarization
from src.agents.corag_agents import CoRAGSubQueryGeneratorAgent, CoRAGSubSolver, CoRAGFinalSolver
from src.agents.simple_agent import SimpleAgent
from src.agents.rewoo_agents import SolverREWOO, WorkerREWOO, PlannerREWOO
from src.agents.maps_agents import ScholarMAPS, SolverMAPS, UserProxyMAPS, ManagerMAPS, AlignerMAPS, CriticMAPS
from src.agents.embedding_agent import EmbeddigAgent
from src.agent_constructor.db import LocalDB, LocalRaptorDB
from src.rag.corag.retriver import CoRAGRetriver
from src.rag.instructrag.retriver import InstructRAGRetriver
from src.rag.raptor.retriver import RaptorRetriver
from src.agent_constructor.chunkers import SimpleChunker, DummyChunker
from src.agent_constructor.filters import LengthFilter
from src.agent_constructor.context_engine import CoRAGContextAssembler


class ComponentFactory:
    def __init__(self):
        self._instances = {}
    
    def create_component(self, config: AnyConfig, dependencies: Dict[str, Any] = None):
        dependencies = dependencies or {}
        
        if isinstance(config, DBConfig):
            return self._create_db(config, dependencies)
        elif isinstance(config, AgentConfig):
            return self._create_agent(config, dependencies)
        elif isinstance(config, RetrieverConfig):
            return self._create_retriever(config, dependencies)
        elif isinstance(config, FilterConfig):
            return self._create_filter(config, dependencies)
        elif isinstance(config, ChunkerConfig):
            return self._create_chunker(config, dependencies)
        elif isinstance(config, ContextAssemblerConfig):
            return self._create_assembler(config, dependencies)
        else:
            raise ValueError(f"Unknown component type: {type(config)}")
    
    def _create_db(self, config: DBConfig, dependencies: Dict):
        if config.type == "local":
            embedding_agent = dependencies.get("embedding_agent")
            chunker = dependencies.get("chunker")
            filter = dependencies.get("filter")
            return LocalDB(
                chunker=chunker,
                embedding_model=embedding_agent,
                filter=filter,
                path_to_db=config.params.get("path_to_db", 'data/docs_database.db'),
                path_to_vector_db=config.params.get("path_to_vector_db", 'data/docs_vector_database'),
                collection_name=config.params.get("collection_name", 'docs'),
            )
        elif config.type == "local_raptor":
            chunker = dependencies.get("chunker")
            return LocalRaptorDB(
                chunker=chunker,
                path_to_db=config.params.get("path_to_db", "data/docs_database.db")
            )
        else:
            raise ValueError(f"Unknown DB type: {config.type}")
    
    def _create_agent(self, config: AgentConfig, dependencies: Dict):
        if config.type == "dummy_planner":
            return DummyPlanner(
                name=config.params.get("name", "dummy_planner"),
                max_subqueries=config.params.get("max_subqueries", 5)
                )
        elif config.type == "dummy_qa":
            return DummyQAAgent(
                name=config.params.get("name", "dummy_qa")
            )
        elif config.type == "dummy_summarization":
            return DummySummarization(
                name=config.params.get("name", "dummy_summarization")
            )
        elif config.type == "tfidf_embedding":
            return TFIDFEmbedding(
                name=config.params.get("name", "tfidf_embedding"),
                vectorizer_path=config.params.get("vectorizer_path", 'data/tfidf_vectorizer.pkl')
            )
        elif config.type == "rewoo_solver":
            return SolverREWOO(
                name=config.params.get("name", "rewoo_solver")
            )
        elif config.type == "rewoo_worker":
            return WorkerREWOO(
                name=config.params.get("name", "rewoo_worker")
            )
        elif config.type == "rewoo_planner":
            return PlannerREWOO(
                name=config.params.get("name", "rewoo_planner")
            )
        elif config.type == "maps_solver":
            return SolverMAPS(
                name=config.params.get("name", "maps_solver")
            )
        elif config.type == "maps_scholar":
            return ScholarMAPS(
                name=config.params.get("name", "maps_scholar")
            )
        elif config.type == "maps_manager":
            return ManagerMAPS(
                name=config.params.get("name", "maps_manager")
            )
        elif config.type == "maps_aligner":
            return AlignerMAPS(
                name=config.params.get("name", "maps_aligner")
            )
        elif config.type == "maps_critic":
            return CriticMAPS(
                name=config.params.get("name", "maps_critic")
            )
        elif config.type == "maps_user_proxy":
            return UserProxyMAPS(
                name=config.params.get("name", "maps_user_proxy")
            )
        elif config.type == "embedding":
            return EmbeddigAgent(
                url=config.params.get("url"),
                model_name=config.params.get("model_name"),
            )
        elif config.type == "corag_sub_generator":
            return CoRAGSubQueryGeneratorAgent(
                url=config.params.get("url"),
                model_name=config.params.get("model_name"),
            )
        elif config.type == "corag_sub_solver":
            return CoRAGSubSolver(
                url=config.params.get("url"),
                model_name=config.params.get("model_name"),
            )
        elif config.type == "corag_final_solver":
            return CoRAGFinalSolver(
                url=config.params.get("url"),
                model_name=config.params.get("model_name"),
            )
        elif config.type == "corag_subq_generator":
            return CoRAGSubQueryGeneratorAgent(
                url=config.params.get("url"),
                model_name=config.params.get("model_name"),
            )
        elif config.type == "simple":
            return SimpleAgent(
                url=config.params.get("url"),
                model_name=config.params.get("model_name"), 
            )
        else:
            raise ValueError(f"Unknown agent type: {config.type}")
    
    def _create_retriever(self, config: RetrieverConfig, dependencies: Dict):
        if config.type == "corag":
            db = dependencies.get("db")
            generator = dependencies.get("generator")
            sub_solver = dependencies.get("sub_solver")
            return CoRAGRetriver(
                name=config.params.get("name", "corag_retriever"),
                data_base=db,
                generator=generator,
                sub_solver=sub_solver,
                max_sub_queries=config.params.get("max_sub_queries")
            )
        elif config.type == "raptor":
            db = dependencies.get("db")
            embedding_agent = dependencies.get("embedding_agent")
            qa_agent = dependencies.get("qa_agent")
            summarization_agent = dependencies.get("summarization_agent")
            return RaptorRetriver(
                name=config.params.get("name", "raptor_retriver"),
                data_base=db,
                path_to_raptor_db=config.params.get("path_to_raptor_db", "data/docs_raptor_database"),
                qa_model=qa_agent,
                summarization_model=summarization_agent,
                embeddig_model=embedding_agent
            )
        elif config.type == "instruct":
            db = dependencies.get("db")
            rationality_agent = dependencies.get("rationality_agent")
            return InstructRAGRetriver(
                name=config.params.get("name", "instruct_rag_retriver"),
                data_base=db,
                rationality_agent=rationality_agent,
            )
        else:
            raise ValueError(f"Unknown retriever type: {config.type}")
    
    def _create_filter(self, config: FilterConfig, dependencies: Dict):
        if config.type == "length":
            return LengthFilter(
                min_len=config.params.get("min_len", 20)
            )

        raise ValueError(f"Unknown filter type: {config.type}")
    
    def _create_chunker(self, config: ChunkerConfig, dependencies: Dict):
        if config.type == "dummy":
            return DummyChunker()
        elif config.type == "simple":
            return SimpleChunker(
                max_chars=config.params.get("max_chars", 1000)
            )
        else:
            raise ValueError(f"Unknown filter type: {config.type}")
        
    def _create_assembler(self, config: ContextAssemblerConfig, dependencies: Dict):
        if config.type == "corag":
            return CoRAGContextAssembler(
                name=config.params.get("name", "corag_context_assembler"),
            )
        else:
            raise ValueError(f"Unknown context assembler type: {config.type}")
