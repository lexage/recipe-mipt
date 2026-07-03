"""Download the CodeMMLU subsets and print examples by index.

Loads the two CodeMMLU sub-datasets used elsewhere in this repo:
  * `fill_in_the_middle` -> ds_middle
  * `code_completion`    -> ds_code

and lets you inspect single examples by their row index.

Login (e.g. `huggingface-cli login`) may be required to access the dataset.

Examples:
    # Print middle example #0 and code example #3
    python download_examples_codemmlu.py --middle 0 --code 3

    # Print several examples from each subset
    python download_examples_codemmlu.py --middle 0 1 2 --code 0 5

    # No indices given -> print the first example of each subset
    python download_examples_codemmlu.py

    # Save the selected examples to a JSON file
    python download_examples_codemmlu.py --middle 0 1 --code 3 --save examples.json
"""

import argparse
import json

import pandas as pd
from datasets import load_dataset


def to_jsonable(value):
    """Convert pandas/numpy values into plain JSON-serializable Python types."""
    if hasattr(value, "tolist"):  # numpy arrays and numpy scalars
        return value.tolist()
    if hasattr(value, "item"):  # remaining numpy scalar types
        return value.item()
    return value


def load_subsets():
    """Load both CodeMMLU subsets as pandas DataFrames."""
    ds_middle = load_dataset("Fsoft-AIC/CodeMMLU", "fill_in_the_middle")["test"].to_pandas()
    ds_code = load_dataset("Fsoft-AIC/CodeMMLU", "code_completion")["test"].to_pandas()
    return ds_middle, ds_code


def print_example(df: pd.DataFrame, index: int, name: str):
    """Pretty-print a single example (row) of a subset by its index."""
    print("=" * 80)
    print(f"[{name}] example #{index}  (out of {len(df)})")
    print("=" * 80)

    if index < 0 or index >= len(df):
        print(f"  index {index} is out of range [0, {len(df)})")
        print()
        return

    row = df.iloc[index]
    for column in df.columns:
        value = row[column]
        print(f"--- {column} ---")
        print(value)
        print()


def example_to_dict(df: pd.DataFrame, index: int, name: str):
    """Return a single example as a JSON-serializable dict, or None if out of range."""
    if index < 0 or index >= len(df):
        return None

    row = df.iloc[index]
    return {
        "subset": name,
        "index": index,
        "data": {column: to_jsonable(row[column]) for column in df.columns},
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download CodeMMLU subsets and print examples by index"
    )
    parser.add_argument(
        "--middle",
        type=int,
        nargs="*",
        default=None,
        help="indices of fill_in_the_middle examples to print",
    )
    parser.add_argument(
        "--code",
        type=int,
        nargs="*",
        default=None,
        help="indices of code_completion examples to print",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="path to a JSON file to save the selected examples to",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ds_middle, ds_code = load_subsets()

    # Default to the first example of each subset when nothing is requested.
    middle_indices = args.middle
    code_indices = args.code
    if middle_indices is None and code_indices is None:
        middle_indices = [0]
        code_indices = [0]
    else:
        middle_indices = middle_indices or []
        code_indices = code_indices or []

    saved = []
    for index in middle_indices:
        print_example(ds_middle, index, "fill_in_the_middle")
        example = example_to_dict(ds_middle, index, "fill_in_the_middle")
        if example is not None:
            saved.append(example)

    for index in code_indices:
        print_example(ds_code, index, "code_completion")
        example = example_to_dict(ds_code, index, "code_completion")
        if example is not None:
            saved.append(example)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(saved, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(saved)} example(s) to {args.save}")


if __name__ == "__main__":
    main()
