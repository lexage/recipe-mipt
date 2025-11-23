from enum import Enum


class ComponentNames(Enum):
    # Agents
    DUMMY_AGENT = "dummy"
    TFIDF_EMBEDDING = "tfidf_embedding"
    REWOO_SOLVER = "rewoo_solver"
    REWOO_WORKER = "rewoo_worker"
    REWOO_PLANNER = "rewoo_planner"
    MAPS_SOLVER = "maps_solver"
    MAPS_SCHOLAR = "maps_scholar"
    MAPS_MANAGER = "maps_manager"
    MAPS_ALIGNER = "maps_aligner"
    MAPS_CRITIC = "maps_critic"
    MAPS_USER_PROXY = "maps_user_proxy"
    EMBEDDING = "embedding"
    CORAG_SUB_GENERATOR = "corag_sub_generator"
    CORAG_SUB_SOLVER = "corag_sub_solver"
    CORAG_FINAL_SOLVER = "corag_final_solver"
    SIMPLE_AGENT = "simple"

    # DBs
    LOCAL_DB = "local"
    LOCAL_RAPTOR_DB = "local_raptor"

    # Retrievers
    CORAG_RETRIEVER = "corag"
    RAPTOR_RETRIEVER = "raptor"
    INSTRUCT_RETRIEVER = "instruct"

    # Filters
    LENGTH_FILTER = "length"

    # Chunkers
    DUMMY_CHUNKER = "dummy"
    SIMPLE_CHUNKER = "simple"

    # Assemblers
    CORAG_CONTEXT_ASSEMBLER = "corag"
