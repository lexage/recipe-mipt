import os
import concurrent.futures as cfuts
import pandas as pd

from typing import Callable, Optional
from tqdm import tqdm

from . import execution
from .dataset import DatasetDS1000




class DS1000:
    """Main class for evaluating models on the DS1000 benchmark.
    
    The DS1000 benchmark tests code generation capabilities across multiple
    Python libraries and perturbation types. Evaluation is performed by
    executing generated code in isolated processes and checking correctness.
    
    Args:
        path: Path to the DS1000 dataset file in .jsonl.gz format.
    
    Example:
        >>> benchmark = DS1000("./data/ds1000.jsonl.gz")
        >>> 
        >>> def my_model_inference(code_prompt):
        ...     # Your model inference logic here
        ...     return generated_code
        >>> 
        >>> results = benchmark.eval(run_method=my_model_inference)
        >>> print(results)
    """
    def __init__(self, path: str):
        # disable tensorflow logging and no GPU
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.dataset = DatasetDS1000(path)


    def eval(self, run_method: Callable, preprocess_method: Optional[Callable] = None):
        """Evaluates a code generation model on the DS1000 benchmark.
        
        Args:
            run_method: Callable that takes a task (or preprocessed task) and 
                       returns generated code as a string.
            preprocess_method: Optional callable to preprocess DataItemDS1000 
                             before passing to run_method. If None, the raw
                             DataItemDS1000 is passed to run_method.
        
        Returns:
            str: A formatted summary string containing:
                - Overall accuracy and count
                - Accuracy per library (NumPy, Pandas, etc.)
                - Accuracy per perturbation type
            
        Note:
            Evaluation uses ProcessPoolExecutor with 16 workers for parallel
            execution. Each code sample is executed with a 120-second timeout.
            
        Example:
            >>> # Simple evaluation with identity functions
            >>> def preprocess(task):
            ...     return task.prompt
            >>> 
            >>> def run_model(prompt):
            ...     return "np.ones(5)"  # Mock model output
            >>> 
            >>> summary = benchmark.eval(
            ...     run_method=run_model,
            ...     preprocess_method=preprocess
            ... )
        """
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
