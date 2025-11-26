from enum import Enum


class ComponentNames(Enum):
    # Agents
    DUMMY_AGENT = "dummy_agent"
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
    EMBEDDING_AGENT = "embedding_agent"
    CORAG_SUB_GENERATOR = "corag_sub_generator"
    CORAG_SUB_SOLVER = "corag_sub_solver"
    CORAG_FINAL_SOLVER = "corag_final_solver"
    SIMPLE_AGENT = "simple_agent"
    RAPTOR_QA_AGENT = "raptor_qa_agent"
    RAPTOR_SUMM_AGENT = "raptor_summ_agent"

    # DBs
    LOCAL_DB = "local_db"
    LOCAL_RAPTOR_DB = "local_raptor_db"

    # Retrievers
    CORAG_RETRIEVER = "corag_retriever"
    RAPTOR_RETRIEVER = "raptor_retriever"
    INSTRUCT_RETRIEVER = "instruct_retriever"

    # Filters
    LENGTH_FILTER = "length_filter"

    # Chunkers
    DUMMY_CHUNKER = "dummy_chunker"
    SIMPLE_CHUNKER = "simple_chunker"

    # Assemblers
    CORAG_CONTEXT_ASSEMBLER = "corag_assembler"
