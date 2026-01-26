import sys
import argparse
import json
import sqlite3
import time
from difflib import SequenceMatcher
from typing import List, Dict, Any
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from src.agent_constructor.core import Chunk
from src.filtering.exactsubstr.main import ExactSubstrFiltrator


class SQLiteDocsDBAdapter:
    def __init__(self, path_to_db: str):
        self.path_to_db = path_to_db

    def get_docs(self) -> List[Chunk]:

        documents = []
        with sqlite3.connect(self.path_to_db) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT d.id, d.filename, d.content, d.file_path, 
                    s.name as section, l.name as library
                FROM documents d
                JOIN sections s ON d.section_id = s.id
                JOIN libraries l ON s.library_id = l.id
                WHERE section == 'user_guide'
            """
            )

            for row in cursor.fetchall():
                documents.append(
                    Chunk(
                        id=row["id"],
                        doc_id=row["id"],
                        text=row["content"],
                        metadata={},
                    )
                )

        return documents


def extract_removed_parts(original: str, filtered: str) -> List[str]:
    """Extract removed substrings using difflib."""
    matcher = SequenceMatcher(None, filtered, original)
    removed_parts = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'insert':
            # This part exists in original but not in filtered → was removed
            removed_parts.append(original[j1:j2])
        elif tag == 'replace':
            # Part was replaced → treat as removal of original segment
            removed_parts.append(original[j1:j2])
    
    return removed_parts


def save_filtered_data(filtrate_documents: List[Chunk], original_docs_by_id: Dict[int, Chunk], output_path: str = "filtered_data.json"):
    """Save filtered documents to a JSON file with original and processed content."""
    data_to_save = []
    for doc in filtrate_documents:
        original_doc = original_docs_by_id.get(doc.id)
        if original_doc:
            data_to_save.append({
                "id": doc.id,
                "original_text": original_doc.text,
                "filtered_text": doc.text,
                "original_length": len(original_doc.text),
                "filtered_length": len(doc.text),
                "removed_substrings": extract_removed_parts(original_doc.text, doc.text),
                "metadata": doc.metadata,
            })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)


def save_stats(
    elapsed_time: float,
    total_initial: int,
    total_filtered: int,
    removed_count: int,
    modified_count: int,  # <-- новая метрика
    stats_per_removed: List[Dict[str, Any]],
    stats_per_kept: List[Dict[str, Any]],
    output_path: str = "filter_stats.json"
):
    """Save filtering statistics to a JSON file."""
    stats = {
        "processing_time_seconds": round(elapsed_time, 2),
        "total_initial_documents": total_initial,
        "total_filtered_documents": total_filtered,
        "removed_documents_count": removed_count,
        "modified_documents_count": modified_count,
        "kept_documents_stats": stats_per_kept,
        "removed_documents_stats": stats_per_removed,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def collect_kept_documents_stats(
    filtrate_documents: List[Chunk],
    original_docs_by_id: Dict[int, Chunk]
) -> tuple[List[Dict[str, Any]], int]:
    """
    Collect statistics for documents that were kept after filtering.
    Returns:
        - list of stats per kept document
        - number of documents whose text was actually modified by the filter
    """
    stats = []
    modified_count = 0

    for doc in filtrate_documents:
        original_doc = original_docs_by_id.get(doc.id)
        if original_doc:
            orig_len = len(original_doc.text)
            filt_len = len(doc.text)
            is_modified = (orig_len != filt_len)

            stats.append({
                "id": doc.id,
                "original_length": orig_len,
                "filtered_length": filt_len,
                "was_modified": is_modified
            })

            if is_modified:
                modified_count += 1

    return stats, modified_count


def collect_removed_documents_stats(
    removed_docs: List[Chunk],
    filtrator: ExactSubstrFiltrator,
    threshold: int
) -> List[Dict[str, Any]]:
    """Collect statistics (including post-processed length) for removed documents."""
    stats = []
    for doc in removed_docs:
        processed_docs = filtrator.apply([doc])
        processed_length = len(processed_docs[0].text) if processed_docs else 0

        stats.append({
            "id": doc.id,
            "original_length": len(doc.text),
            "postprocessed_length": processed_length,
            "reason": f"Post-processed length ({processed_length}) < threshold ({threshold})"
        })
    return stats


def print_summary_report(
    elapsed_time: float,
    total_initial: int,
    total_filtered: int,
    removed_count: int,
    modified_count: int,
    stats_per_removed: List[Dict[str, Any]]
):
    """Print a human-readable summary of the filtering results."""
    print("\n" + "=" * 50)
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Total input documents: {total_initial}")
    print(f"Documents retained after filtering: {total_filtered}")
    print(f"Documents removed: {removed_count}")
    print(f"Documents with modified content: {modified_count} (out of {total_filtered} kept)")
    print("=" * 50)

    if removed_count > 0:
        print("\nExamples of removed documents (first 3):")
        for i, stat in enumerate(stats_per_removed[:3]):
            print(f"  {i+1}. ID={stat['id']}, original length={stat['original_length']}, "
                  f"post-processed length={stat['postprocessed_length']}, reason: {stat['reason']}")


def process_filtering_results(
    documents: List[Chunk],
    filtrate_documents: List[Chunk],
    filtrator: ExactSubstrFiltrator,
    threshold: int,
    elapsed_time: float,
    output_filtered: str,
    output_stats: str
):
    """
    Process filtering results: compute statistics, save outputs, and print summary.
    """
    # Build index for fast lookup of original documents
    original_docs_by_id = {doc.id: doc for doc in documents}

    # Analyze kept documents
    kept_stats, modified_count = collect_kept_documents_stats(filtrate_documents, original_docs_by_id)  # ✅

    # Identify and analyze removed documents
    kept_ids = {doc.id for doc in filtrate_documents}
    removed_docs = [doc for doc in documents if doc.id not in kept_ids]
    removed_stats = collect_removed_documents_stats(removed_docs, filtrator, threshold)  # ✅

    # Compute high-level metrics
    total_initial = len(documents)
    total_filtered = len(filtrate_documents)
    removed_count = len(removed_docs)

    # Save filtered data
    save_filtered_data(filtrate_documents, original_docs_by_id, output_filtered)
    print(f"Filtered data saved to: {output_filtered}")

    # Save detailed statistics
    save_stats(  # ✅
        elapsed_time=elapsed_time,
        total_initial=total_initial,
        total_filtered=total_filtered,
        removed_count=removed_count,
        modified_count=modified_count,
        stats_per_removed=removed_stats,
        stats_per_kept=kept_stats,
        output_path=output_stats
    )
    print(f"Filtering statistics saved to: {output_stats}")

    # Print human-readable summary
    print_summary_report(
        elapsed_time=elapsed_time,
        total_initial=total_initial,
        total_filtered=total_filtered,
        removed_count=removed_count,
        modified_count=modified_count,
        stats_per_removed=removed_stats
    )


def main():
    parser = argparse.ArgumentParser(
        description="Filter DS1000 dataset using the exact substring filtering method from the paper.",
    )

    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        default="/workspace/proj/grant/data/docs_database_dedup.db",
        help="Path to the SQLite database file",
    )
    
    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=60,
        help="Threshold length for removing short or duplicate-like documents",
    )

    parser.add_argument(
        "--output_filtered",
        type=str,
        default="filtered_data.json",
        help="Output path for the filtered documents in JSON format",
    )

    parser.add_argument(
        "--output_stats",
        type=str,
        default="filter_stats.json",
        help="Output path for filtering statistics in JSON format",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit the number of documents processed (useful for testing)",
    )
    
    args = parser.parse_args()

    data_path = args.data_path
    threshold = args.threshold
    output_filtered = args.output_filtered
    output_stats = args.output_stats
    limit = args.limit

    sql_client = SQLiteDocsDBAdapter(data_path)
    documents = sql_client.get_docs()
    if limit is not None:
        documents = documents[:limit]
    
    print(f"Loaded {len(documents)} documents.")
    
    filtrator = ExactSubstrFiltrator(
        threshold=threshold,
        enable_bytes=True,
        enable_tokenizer=False,
        tokenizer=None
    )
    
    start_time = time.time()
    filtrate_documents = filtrator.apply(documents)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Filtering completed in {elapsed_time:.2f} seconds.")
    print(f"Kept {len(filtrate_documents)} out of {len(documents)} documents after filtering.")
    
    process_filtering_results(
        documents=documents,
        filtrate_documents=filtrate_documents,
        filtrator=filtrator,
        threshold=args.threshold,
        elapsed_time=elapsed_time,
        output_filtered=args.output_filtered,
        output_stats=args.output_stats
    )
    

if __name__ == "__main__":
    main()
