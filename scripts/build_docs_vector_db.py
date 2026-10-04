#!/usr/bin/env python3
"""Build the local Qdrant documentation index through SimplePipeline initialization."""

import argparse
from pathlib import Path

from src.pipelines.configs import ConfigLoader
from src.pipelines.pipeline_builder import PipelineBuilder


DEFAULT_CONFIG = (
    "pipeline_configs/react_no_sgr/vector_db_build/build_vector_db.yaml"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the documentation vector DB used by ReAct RAG configs."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to the SIMPLE pipeline config used to build the vector DB.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    print(f"Building vector DB from config: {config_path}")
    pipeline_config = ConfigLoader().load_from_yaml(path_to_cfg=config_path)

    pipeline = PipelineBuilder().build(pipeline_config)
    try:
        print("Vector DB build completed.")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
