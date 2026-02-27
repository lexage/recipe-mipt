import argparse
import logging

from src.benchmarks import DS1000, DataItemDS1000
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
        "-d", "--dataset", default="data/ds1000/ds1000.jsonl.gz", help="Path to DS1000 dataset"
    )
    parser.add_argument(
        "-s", "--save_path", default="results", help="Path to where to save results"
    )
    parser.add_argument(
        "-e", "--experiment", default=None, help="Experiment directory to continue"
    )
    parser.add_argument(
        "-n", "--num_workers", default=4, help="Number of workers"
    )
    return parser.parse_args()


def main():

    args = parse_args()

    pipeline = PipelineBuilder().build(load(args.config))

    bench = DS1000(dataset_path=args.dataset)

    def run_pipeline(task: DataItemDS1000):
        return pipeline.run(task.prompt)

    bench.eval(
        run_method=run_pipeline,
        save_path=args.save_path,
        continue_exp=args.experiment,
        num_workers=int(args.num_workers)
        )


if __name__ == "__main__":
    main()
