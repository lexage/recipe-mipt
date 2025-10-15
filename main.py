from src.benchmarks.ds1000 import DS1000    


def dummy_preprocess(task):
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
