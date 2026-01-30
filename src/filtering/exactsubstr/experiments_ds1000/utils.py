import json
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from src.agent_constructor.core import Chunk
from src.filtering.exactsubstr.main import ExactSubstrFiltrator


def extract_removed_parts(original: str, filtered: str) -> List[str]:
    """Extract removed substrings using difflib."""
    matcher = SequenceMatcher(None, filtered, original)
    removed_parts = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            # This part exists in original but not in filtered → was removed
            removed_parts.append(original[j1:j2])
        elif tag == "replace":
            # Part was replaced → treat as removal of original segment
            removed_parts.append(original[j1:j2])

    return removed_parts


def save_filtered_data(
    filtrate_chunks: List[Chunk],
    original_docs_by_id: Dict[int, Chunk],
    output_path: str = "filtered_data.json",
):
    """Save filtered chunks to a JSON file with original and processed content."""
    data_to_save = []
    for doc in filtrate_chunks:
        original_doc = original_docs_by_id.get(doc.id)
        if original_doc:
            data_to_save.append(
                {
                    "id": doc.id,
                    "original_text": original_doc.text,
                    "filtered_text": doc.text,
                    "original_length": len(original_doc.text),
                    "filtered_length": len(doc.text),
                    "removed_substrings": extract_removed_parts(
                        original_doc.text, doc.text
                    ),
                    "metadata": doc.metadata,
                }
            )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)


def save_stats(
    elapsed_time: float,
    total_initial: int,
    total_filtered: int,
    removed_count: int,
    modified_count: int,
    stats_per_removed: List[Dict[str, Any]],
    stats_per_kept: List[Dict[str, Any]],
    output_path: str = "filter_stats.json",
):
    """Save filtering statistics to a JSON file."""
    stats = {
        "processing_time_seconds": round(elapsed_time, 2),
        "total_initial_chunks": total_initial,
        "total_filtered_chunks": total_filtered,
        "removed_chunks_count": removed_count,
        "modified_chunks_count": modified_count,
        "kept_chunks_stats": stats_per_kept,
        "removed_chunks_stats": stats_per_removed,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def collect_kept_chunks_stats(
    filtrate_chunks: List[Chunk], original_docs_by_id: Dict[int, Chunk]
) -> tuple[List[Dict[str, Any]], int]:
    """
    Collect statistics for chunks that were kept after filtering.
    Returns:
        - list of stats per kept chunk
        - number of chunks whose text was actually modified by the filter
    """
    stats = []
    modified_count = 0

    for doc in filtrate_chunks:
        original_doc = original_docs_by_id.get(doc.id)
        if original_doc:
            orig_len = len(original_doc.text)
            filt_len = len(doc.text)
            is_modified = orig_len != filt_len

            stats.append(
                {
                    "id": doc.id,
                    "original_length": orig_len,
                    "filtered_length": filt_len,
                    "was_modified": is_modified,
                }
            )

            if is_modified:
                modified_count += 1

    return stats, modified_count


def collect_removed_chunks_stats(
    removed_docs: List[Chunk], filtrator: ExactSubstrFiltrator, threshold: int
) -> List[Dict[str, Any]]:
    """Collect statistics (including post-processed length) for removed chunks."""
    stats = []
    for doc in removed_docs:
        processed_docs = filtrator.apply([doc])
        processed_length = len(processed_docs[0].text) if processed_docs else 0

        stats.append(
            {
                "id": doc.id,
                "original_length": len(doc.text),
                "postprocessed_length": processed_length,
                "reason": f"Post-processed length ({processed_length}) < threshold ({threshold})",
            }
        )
    return stats


def print_summary_report(
    elapsed_time: float,
    total_initial: int,
    total_filtered: int,
    removed_count: int,
    modified_count: int,
    stats_per_removed: List[Dict[str, Any]],
):
    """Print a human-readable summary of the filtering results."""
    print("\n" + "=" * 50)
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Total input chunks: {total_initial}")
    print(f"Chunks retained after filtering: {total_filtered}")
    print(f"Chunks removed: {removed_count}")
    print(
        f"Chunks with modified content: {modified_count} (out of {total_filtered} kept)"
    )
    print("=" * 50)

    if removed_count > 0:
        print("\nExamples of removed chunks (first 3):")
        for i, stat in enumerate(stats_per_removed[:3]):
            print(
                f"  {i+1}. ID={stat['id']}, original length={stat['original_length']}, "
                f"post-processed length={stat['postprocessed_length']}, reason: {stat['reason']}"
            )


def process_filtering_results(
    chunks: List[Chunk],
    filtrate_chunks: List[Chunk],
    filtrator: ExactSubstrFiltrator,
    threshold: int,
    elapsed_time: float,
    output_filtered: str,
    output_stats: str,
):
    """
    Process filtering results: compute statistics, save outputs, and print summary.
    """
    original_docs_by_id = {doc.id: doc for doc in chunks}

    kept_stats, modified_count = collect_kept_chunks_stats(
        filtrate_chunks, original_docs_by_id
    )  # ✅

    kept_ids = {doc.id for doc in filtrate_chunks}
    removed_docs = [doc for doc in chunks if doc.id not in kept_ids]
    removed_stats = collect_removed_chunks_stats(
        removed_docs, filtrator, threshold
    )  # ✅

    total_initial = len(chunks)
    total_filtered = len(filtrate_chunks)
    removed_count = len(removed_docs)

    save_filtered_data(filtrate_chunks, original_docs_by_id, output_filtered)
    print(f"Filtered data saved to: {output_filtered}")

    save_stats(  # ✅
        elapsed_time=elapsed_time,
        total_initial=total_initial,
        total_filtered=total_filtered,
        removed_count=removed_count,
        modified_count=modified_count,
        stats_per_removed=removed_stats,
        stats_per_kept=kept_stats,
        output_path=output_stats,
    )
    print(f"Filtering statistics saved to: {output_stats}")

    print_summary_report(
        elapsed_time=elapsed_time,
        total_initial=total_initial,
        total_filtered=total_filtered,
        removed_count=removed_count,
        modified_count=modified_count,
        stats_per_removed=removed_stats,
    )
