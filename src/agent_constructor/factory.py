from typing import Dict, Any
from pydantic import BaseModel

from src.agent_constructor.configs import AgentConfig, DBConfig, FilterConfig, RetrieverConfig, ChunkerConfig, AnyConfig

from src.agents.tfidf_agent import TFIDFEmbedding
from src.agents.dummy_agents import DummyPlanner, DummyQAAgent, DummySummarization
from src.agent_constructor.db import LocalDB
from src.rag.corag.retriver import CoRAGRetriver
from src.agent_constructor.chunkers import SimpleChunker, DummyChunker


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
        else:
            raise ValueError(f"Unknown component type: {type(config)}")
    
    def _create_db(self, config: DBConfig, dependencies: Dict):
        if config.type == "local":
            embedding_agent = dependencies.get("embedding_agent")
            chunker = dependencies.get("chunker")
            return LocalDB(
                chunker=chunker,
                embedding_model=embedding_agent,
                path_to_db=config.params.get("path_to_db", 'data/docs_database.db'),
                path_to_vector_db=config.params.get("path_to_vector_db", 'data/docs_vector_database'),
                collection_name=config.params.get("collection_name", 'docs'),
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
        else:
            raise ValueError(f"Unknown agent type: {config.type}")
    
    def _create_retriever(self, config: RetrieverConfig, dependencies: Dict):
        if config.type == "corag":
            db = dependencies.get("db")
            planner_agent = dependencies.get("planner_agent")
            return CoRAGRetriver(
                name=config.params.get("name", "corag_retriever"),
                data_base=db,
                planner_agent=planner_agent,
            )
        else:
            raise ValueError(f"Unknown retriever type: {config.type}")
    
    def _create_filter(self, config: FilterConfig, dependencies: Dict):
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
