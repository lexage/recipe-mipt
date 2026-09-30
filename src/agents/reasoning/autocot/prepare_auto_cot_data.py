import json
from src.benchmarks.ds1000.dataset import DatasetDS1000


data_path = "/workspace/proj/grant/recipe-mipt/data/ds1000/ds1000.jsonl.gz"
dataset = DatasetDS1000(data_path)

all_examples = []

for item in dataset:
    example = {}
    example["question"] = item.prompt
    example["code"] = item.reference_code
    all_examples.append(example)

with open("/workspace/proj/grant/recipe-mipt/data/ds1000_auto_cot.json", "w", encoding="utf-8") as f:
    json.dump(all_examples, f, indent=4, ensure_ascii=False)


# with open("/workspace/proj/grant/recipe-mipt/data/ds1000_auto_cot.json", "r") as f:
#     data = json.load(f)
#     print(data)