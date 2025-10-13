import gzip
import json
import os
import concurrent.futures as cfuts
import pandas as pd

from dataclasses import dataclass
from tqdm import tqdm

import execution

# disable tensorflow logging and no GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@dataclass
class DataItemDS1000:
    prompt: str
    reference_code: str
    metadata: dict
    code_context: str

    @classmethod
    def from_dict(cls, data: dict) -> 'DataItemDS1000':
        return cls(
            prompt=data['prompt'],
            reference_code=data['reference_code'],
            metadata=data['metadata'],
            code_context=data['code_context'],
        )


class DatasetDS1000:
    def __init__(self, path):
        self.path = path
    
    def preprocess(self, line: str) -> DataItemDS1000:
        return DataItemDS1000.from_dict(json.loads(line))
    
    def __iter__(self):
        with gzip.open(self.path, "rt") as f:
            for line in f:
                yield self.preprocess(line)



class DS1000:
    def __init__(self, path: str):
        self.dataset = DatasetDS1000(path)


    def eval(self, run_method, preprocess_method = None):
        ds1000_results = []

        with cfuts.ProcessPoolExecutor(
            max_workers=16
            ) as executor:
            
            futs = []

            for task in self.dataset:

                if preprocess_method:
                    preprocess_task = preprocess_method(task)
                    result = run_method(preprocess_task)
                else:
                    result = run_method(task)

                test_program = (
                    task.code_context + '\n'
                    + f'code = {result}\n'
                    + 'test_execution(code)\n'
                    + ('test_string(code)\n'  if 'test_string(' in task.code_context  else '\n')
                )

                futs.append(executor.submit(execution.check_correctness, test_program, timeout=120, metadata=task.metadata))

            for f in tqdm(cfuts.as_completed(futs), total=len(futs)):
                eval_result, metadata = f.result()
                eval_result['score'] = 1 if eval_result['passed'] else 0
                eval_result['library'] = metadata['library']
                eval_result['perturbation_type'] = metadata['perturbation_type']
                ds1000_results.append(eval_result)
            
        df_res = pd.DataFrame.from_records(ds1000_results)
        pd.set_option('display.precision', 3)
        summary = df_res.agg({'score': ['count', 'mean']}).to_string()
        summary += '\n' + df_res[['library', 'score']].groupby('library').agg({'score': ['count', 'mean']}).to_string()
        summary += '\n' + df_res[['perturbation_type', 'score']].groupby('perturbation_type').agg({'score': ['count', 'mean']}).to_string()
        return summary


if __name__ == "__main__":
    
    def dummy_preprocess(task: DataItemDS1000):
        return task.reference_code

    def dummy_run(task):
        return task
    
    benchmark = DS1000("./data/ds1000.jsonl.gz")
    
    summary = benchmark.eval(
        run_method=dummy_run, 
        preprocess_method=dummy_preprocess
    )

    with open(f'results/test-result.txt', 'w') as f:
        f.write(summary)
