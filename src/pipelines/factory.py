from typing import Dict, Any

from src.pipelines.registry import ComponentInfo

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
    
    def create_component(
            self, 
            component_info: ComponentInfo, 
            available_dependencies: Dict[str, Any] = None, 
            config_params = Dict[str, Any]
            ):
        
        params = {}
        
        for param_name, param_info in component_info.params.items():
                
            if param_info.is_dependency:
                params[param_name] = available_dependencies[param_name]

            elif param_name in config_params:
                params[param_name] = config_params[param_name]
            
            elif param_info.has_default:
                continue

            else:
                raise ValueError(
                    f"Required parameter '{param_name}' not found for {component_info.class_}. "
                    f"Available: params={list(config_params)}, "
                )
        
        return component_info.class_(**params)
