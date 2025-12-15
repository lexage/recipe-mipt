from db import InMemoryDB
from pipeline import AgentPipeline
from context_engine import SimpleContextAssembler
from pipeline import PipelineConfig, SimpleCritic, SimplePlanner, MockStudent
from chunkers import SimpleChunker

def build_minimal_pipeline() -> AgentPipeline:
    db = InMemoryDB()
    chunker = SimpleChunker(max_chars=512)
    # Example ingestion: convert docs to chunks and store
    # This is left to examples.py in a full repo; here we only assemble pipeline

    context_assembler = SimpleContextAssembler()

    cfg = PipelineConfig(planner=SimplePlanner(), critic=SimpleCritic(), student=MockStudent())
    return AgentPipeline(db=db, context_assembler=context_assembler, cfg=cfg)