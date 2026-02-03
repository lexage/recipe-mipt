from pydantic_settings import BaseSettings
from typing import TypeAlias


class OutputConfig(BaseSettings):
    output_dir: str = './src/filtering/neardup/output'
    skip_filtering: bool = False
    clean_cache: bool = False
    save_clusters: bool = False
    keep_index_column: bool = False
    keep_cluster_column: bool = False


OutputConfigType: TypeAlias = OutputConfig