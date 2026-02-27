import argparse
import logging

from src.pipelines.pipeline_builder import PipelineBuilder
from src.pipelines.configs import ConfigLoader

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description="Run pipeline with config")
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help=f'Path to config *.yaml file',
    )
    parser.add_argument(
        "-q", "--query", default="What is pandas?", help="Query to process"
    )
    return parser.parse_args()


def main():

    args = parse_args()

    pipeline_config = ConfigLoader().load_from_yaml(args.config)
    pipeline = PipelineBuilder().build(pipeline_config)
    
    result = pipeline.run(args.query)
    print(result)


if __name__ == "__main__":
    main()
