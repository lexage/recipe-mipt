import json
import logging
import subprocess
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import List, Optional

from swebench.harness.utils import (
    str2bool,
    optional_str,
)

from swebench.inference.run_agent_pipeline import main as run_inference
from swebench.harness.run_evaluation import main as run_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(
    # Inference arguments
    method_name: str,
    dataset_path: str,
    run_id: str,
    split: str = "test",
    output_dir: str = "./outputs",
    min_len: int = None,
    max_len: int = None,
    shard_id: int = None,
    num_shards: int = None,
    
    # Evaluation arguments
    instance_ids: Optional[List[str]] = None,
    predictions_path: Optional[str] = None,
    max_workers: int = 4,
    force_rebuild: bool = False,
    cache_level: str = "env",
    clean: bool = False,
    open_file_limit: int = 4096,
    timeout: int = 1800,
    namespace: str = "swebench",
    rewrite_reports: bool = False,
    modal: bool = False,
    instance_image_tag: str = "latest",
    env_image_tag: str = "latest",
    report_dir: str = ".",
):
    """
    Run full pipeline: generate predictions and evaluate them
    """

    logger.info("Starting inference phase...")
    
    if predictions_path is None:
        from swebench.inference.run_agent_pipeline import get_output_file
        predictions_path = get_output_file(
            output_dir=output_dir,
            method_name=method_name,
            dataset_path=dataset_path,
            split=split,
            min_len=min_len,
            max_len=max_len,
            shard_id=shard_id,
            num_shards=num_shards,
        )
        logger.info(f"Predictions will be saved to: {predictions_path}")
    
    inference_output_file = run_inference(
        method_name=method_name,
        dataset_path=dataset_path,
        split=split,
        output_dir=output_dir,
        min_len=min_len,
        max_len=max_len,
        shard_id=shard_id,
        num_shards=num_shards,
    )
    
    logger.info(f"Inference completed. Predictions saved to: {inference_output_file}")
    
    logger.info("Starting evaluation phase...")
    
    evaluation_result = run_evaluation(
        dataset_name=dataset_path,
        split=split,
        instance_ids=instance_ids,
        predictions_path=str(inference_output_file),
        max_workers=max_workers,
        force_rebuild=force_rebuild,
        cache_level=cache_level,
        clean=clean,
        open_file_limit=open_file_limit,
        run_id=run_id if run_id else f"{method_name}_{dataset_path.replace('/', '__')}",
        timeout=timeout,
        namespace=namespace,
        rewrite_reports=rewrite_reports,
        modal=modal,
        instance_image_tag=instance_image_tag,
        env_image_tag=env_image_tag,
        report_dir=report_dir,
    )
    
    logger.info("Evaluation completed!")
    return evaluation_result


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run evaluation harness for the given dataset and predictions.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--method_name",
        type=str,
        required=True,
        help="Method name",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset or hf dataset name",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to use"
    )
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument(
        "--min_len",
        type=int,
        default=None,
        help="Minimum length of input sequences to include",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="Maximum length of input sequences to include",
    )
    parser.add_argument(
        "--shard_id", type=int, default=None, help="ID of the shard to load"
    )
    parser.add_argument(
        "--num_shards", type=int, default=None, help="Total number of shards"
    )
    

    parser.add_argument(
        "-i",
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated)",
    )
    parser.add_argument(
        "-p",
        "--predictions_path",
        type=optional_str,
        default=None,
        help="Path to predictions file - if 'gold', uses gold predictions",
    )

    # Local execution args
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of workers (should be <= 75%% of CPU cores)",
    )
    parser.add_argument(
        "--open_file_limit", type=int, default=4096, help="Open file limit"
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=1_800,
        help="Timeout (in seconds) for running tests for each instance",
    )
    parser.add_argument(
        "--force_rebuild",
        type=str2bool,
        default=False,
        help="Force rebuild of all images",
    )
    parser.add_argument(
        "--cache_level",
        type=str,
        choices=["none", "base", "env", "instance"],
        help="Cache level - remove images above this level",
        default="env",
    )
    # if clean is true then we remove all images that are above the cache level
    # if clean is false, we only remove images above the cache level if they don't already exist
    parser.add_argument(
        "--clean", type=str2bool, default=False, help="Clean images above cache level"
    )
    parser.add_argument(
        "-id", "--run_id", type=str, required=True, help="Run ID - identifies the run"
    )
    parser.add_argument(
        "-n",
        "--namespace",
        type=optional_str,
        default="swebench",
        help='Namespace for images. (use "none" to use no namespace)',
    )
    parser.add_argument(
        "--instance_image_tag", type=str, default="latest", help="Instance image tag"
    )
    parser.add_argument(
        "--env_image_tag", type=str, default="latest", help="Environment image tag"
    )
    parser.add_argument(
        "--rewrite_reports",
        type=str2bool,
        default=False,
        help="Doesn't run new instances, only writes reports for instances with existing test outputs",
    )
    parser.add_argument(
        "--report_dir", type=str, default=".", help="Directory to write reports to"
    )

    # Modal execution args
    parser.add_argument("--modal", type=str2bool, default=False, help="Run on Modal")

    args = parser.parse_args()
    main(**vars(args))