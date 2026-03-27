import argparse
import logging
import os
import glob
import time

from pathlib import Path

from src.benchmarks import DS1000, DataItemDS1000
from src.pipelines.pipeline_builder import PipelineBuilder
from src.pipelines.configs import ConfigLoader
from src.utils.loggers import create_logging


logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description="Run pipeline with config")
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help=f'Path to config dir',
    )
    parser.add_argument(
        "-d", "--dataset", default="/workspace/proj/grant/recipe-mipt/data/ds1000/ds1000.jsonl.gz", help="Path to DS1000 dataset"
    )
    parser.add_argument(
        "-s", "--save_path", default="results", help="Path to where to save results"
    )
    parser.add_argument(
        "-n", "--num_workers", default=4, help="Number of workers"
    )
    return parser.parse_args()


def main():

    args = parse_args()

    config_dir = Path(args.config)
    config_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))

    bench = DS1000(dataset_path=args.dataset)

    for idx, config_path in enumerate(config_files):
        
        print(f"- processing file {idx+1}/{len(config_files)}")
        print(f"\t - config: {config_path.name}")
        
        config_start = time.time()

        pipeline_config = ConfigLoader().load_from_yaml(
            path_to_cfg=config_path
        )
        
        if pipeline_config.logs_path:
            create_logging(
                log_filename=pipeline_config.logs_path,
                tag=config_path.stem
                )

        pipeline = PipelineBuilder().build(pipeline_config)
    
        def run_pipeline(task: DataItemDS1000):
            task_start = time.time()
            result = pipeline.run(task.prompt)
            task_time = time.time() - task_start
            logging.info(f"TASK\t{task.metadata.get('problem_id', 'N/A')}\t{task_time:.3f}s")
            return result
    
        bench.eval(
            run_method=run_pipeline,
            save_path=os.path.join(args.save_path, config_path.stem),
            num_workers=int(args.num_workers)
            )
        
        config_time = time.time() - config_start
        logging.info(f"CONFIG\t{config_path.name}\t{config_time:.3f}s")


if __name__ == "__main__":
    main()