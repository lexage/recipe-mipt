from typing import Dict, Any

from src.pipelines.configs import AgentConfig, DBConfig, FilterConfig, RetrieverConfig, ChunkerConfig, ContextAssemblerConfig, AnyConfig
from src.pipelines.registry import COMPONENT_REGISTRY, create_component_automatically

from src.agent_constructor.db import LocalDB, LocalRaptorDB
from src.agent_constructor.chunkers import SimpleChunker, DummyChunker
from src.agent_constructor.filters import LengthFilter
from src.agent_constructor.context_engine import CoRAGContextAssembler

from src.agents.tfidf_agent import TFIDFEmbedding
from src.agents.dummy_agents import DummyAgent
from src.agents.corag_agents import CoRAGSubQueryGeneratorAgent, CoRAGSubSolver, CoRAGFinalSolver
from src.agents.simple_agent import SimpleAgent
from src.agents.rewoo_agents import SolverREWOO, WorkerREWOO, PlannerREWOO
from src.agents.maps_agents import ScholarMAPS, SolverMAPS, UserProxyMAPS, ManagerMAPS, AlignerMAPS, CriticMAPS
from src.agents.embedding_agent import EmbeddigAgent

from src.rag.corag.retriever import CoRAGRetriever
from src.rag.instructrag.retriever import InstructRAGRetriever
from src.rag.raptor.retriever import RaptorRetriever


class ComponentFactory:
    
    def create_component(self, config: AnyConfig, dependencies: Dict[str, Any] = None):
        config_type = type(config)

        if config_type in COMPONENT_REGISTRY:
            registry = COMPONENT_REGISTRY[config_type]
            if config.type in registry:
                component_info = registry[config.type]
                component_class = component_info['class']
                dependencies_names = component_info['dependencies']
                
                return create_component_automatically(
                    config, 
                    dependencies, 
                    component_class, 
                    dependencies_names
                )
        raise ValueError(f"Unknown component type: {config.type}")
    
