import pickle
from pathlib import Path
from typing import Any, List
from typing import cast

import sys
import os
sys.path.insert(0, os.getcwd())
from operator import attrgetter

from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]
from datasets import disable_progress_bars  # pyright: ignore[reportMissingTypeStubs]
from datasets import enable_progress_bars  # pyright: ignore[reportMissingTypeStubs]
from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_dataset as hf_load_dataset,  # pyright: ignore[reportUnknownVariableType]
)
from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_from_disk as hf_load_from_disk,  # pyright: ignore[reportUnknownVariableType]
)

from src.filtering.neardup.config import Config
from src.filtering.neardup.config.io import LocalInputConfig
from src.filtering.neardup.config.io import OutputConfig
from src.filtering.neardup.config.io.input_configs import LocalHFDatasetInputConfig

from src.filtering.neardup.utils.logger import log
from src.filtering.neardup.utils.progress import use_tqdm

# from src.agent_constructor.core import Document
from src.agent_constructor.core import Chunk


class InvalidDatasetTypeError(Exception):
    def __init__(self, data_type: type) -> None:
        super().__init__(f"Expecting Dataset object, loaded {data_type} instead")


def load_dataset(
    chunks: List[Chunk],
    config: Config
) -> Dataset:
    dataset_dict = {
        "id": list(map(attrgetter('id'), chunks)),
        "doc_id": list(map(attrgetter('doc_id'), chunks)),
        "text": list(map(attrgetter('text'), chunks)),
        "tokens": list(map(attrgetter('tokens'), chunks)),
        "metadata": list(map(attrgetter('metadata'), chunks)),
    }
    
    ds = Dataset.from_dict(dataset_dict)
    _INTERNAL_INDEX_COLUMN = config.algorithm.internal_index_column
    ds = ds.map(  # pyright: ignore[reportUnknownMemberType]
        lambda _, i: {_INTERNAL_INDEX_COLUMN: i},  # pyright: ignore[reportUnknownLambdaType]
        with_indices=True,
        num_proc=config.algorithm.num_proc,
        desc="Indexing",
    )
    return cast(Dataset, ds)
    

def save_dataset(config: Config, *, final_data: Dataset, clusters: dict[int, int], **kwargs: Any) -> None:  # pyright: ignore[reportExplicitAny, reportAny, reportUnusedParameter]
    """Save the dataset to disk."""
    # Create output directory if it doesn't exist
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.output.save_clusters:
        if not config.output.keep_index_column:
            log.warning("Saving clusters requires `--keep-index-column`, turning it on")
            config.output.keep_index_column = True
        with open(output_dir / "clusters.pickle", "wb") as f:
            pickle.dump(clusters, f, protocol=pickle.HIGHEST_PROTOCOL)

    match config.output:
        case OutputConfig():
            columns_to_remove: set[str] = {
                config.algorithm.internal_index_column,
                config.algorithm.cluster_column,
            }
            if config.output.keep_index_column and config.algorithm.internal_index_column in columns_to_remove:
                columns_to_remove.remove(config.algorithm.internal_index_column)
            if (
                config.output.keep_cluster_column or config.output.save_clusters
            ) and config.algorithm.cluster_column in columns_to_remove:
                columns_to_remove.remove(config.algorithm.cluster_column)
            if columns_to_remove:
                columns_to_remove_filtered = [col for col in columns_to_remove if col in final_data.column_names]
                if columns_to_remove_filtered:
                    final_data = final_data.remove_columns(columns_to_remove_filtered)

            final_data.save_to_disk(config.output.output_dir, num_proc=config.algorithm.num_proc)  # pyright: ignore[reportUnknownMemberType]