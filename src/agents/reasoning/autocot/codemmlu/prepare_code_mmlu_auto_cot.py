import json
from datasets import load_dataset

ds_code = load_dataset("Fsoft-AIC/CodeMMLU", "code_completion")["test"].to_pandas()
ds_middle = load_dataset("Fsoft-AIC/CodeMMLU", "fill_in_the_middle")["test"].to_pandas()

# all_examples = []
code_examples = []
middle_examples = []

# Буквы для разметки вариантов (хватит даже если вариантов будет больше 4)
LETTERS = ["A", "B", "C", "D", "E", "F", "G"]

# Обработка code_completion
for _, row in ds_code.iterrows():
    # Собираем только те варианты, которые реально есть в row["choices"]
    solutions = [f"Solution {LETTERS[i]}: {choice}" for i, choice in enumerate(row["choices"])]
    solutions_str = "\n".join(solutions)
    
    question_with_options = f"Problem: {row['question']}\n{solutions_str}"
    
    example = {
        "question": question_with_options,  
        "code": row["answer"],              
        "options": row["choices"].tolist() if hasattr(row["choices"], "tolist") else row["choices"],
        "answer": row["answer"],
        "task_id": row["task_id"],
        "type": "code_completion"
    }
    code_examples.append(example)

# Обработка fill_in_the_middle
for _, row in ds_middle.iterrows():
    full_question = row["question"] + " " + row["problem_description"]
    
    # Собираем варианты динамически для middle
    solutions = [f"Solution {LETTERS[i]}: {choice}" for i, choice in enumerate(row["choices"])]
    solutions_str = "\n".join(solutions)
    
    question_with_options = f"Problem: {full_question}\n{solutions_str}"
    
    example = {
        "question": question_with_options,
        "code": row["answer"],
        "options": row["choices"].tolist() if hasattr(row["choices"], "tolist") else row["choices"],
        "answer": row["answer"],
        "task_id": row["task_id"],
        "type": "fill_in_the_middle"
    }
    middle_examples.append(example)

# Сохранение
with open("/workspace/proj/grant_codemmlu/recipe-mipt/src/agents/reasoning/autocot/codemmlu/code_mmlu_auto_cot_code.json", "w", encoding="utf-8") as f:
    json.dump(code_examples, f, indent=4, ensure_ascii=False)

print(f"Saved {len(code_examples)} examples")

with open("/workspace/proj/grant_codemmlu/recipe-mipt/src/agents/reasoning/autocot/codemmlu/code_mmlu_auto_cot_middle.json", "w", encoding="utf-8") as f:
    json.dump(middle_examples, f, indent=4, ensure_ascii=False)

print(f"Saved {len(middle_examples)} examples")