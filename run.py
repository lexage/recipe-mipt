import argparse
import logging

from src.pipelines.pipeline_builder import PipelineBuilder
from pipeline_configs import available, load

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


def parse_args():
    configs = available()
    parser = argparse.ArgumentParser(description="Run pipeline with config")
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help=f'Config name. Available: {", ".join(configs)}',
    )
    parser.add_argument(
        "-q", "--query", default="What is pandas?", help="Query to process"
    )
    return parser.parse_args()


def main():

    args = parse_args()

    pipeline = PipelineBuilder().build(load(args.config))
    result = pipeline.run(args.query)
    print(result)


if __name__ == "__main__":
    main()
