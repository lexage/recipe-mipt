import os
import re
import json
import concurrent.futures as cfuts

from datetime import datetime
from typing import Callable, Optional
from tqdm import tqdm

from . import execution
from .dataset import DatasetDS1000
from .data_types import ResultsDS1000


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
    """

    def __init__(self, dataset_path: str):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.dataset = DatasetDS1000(dataset_path)

    @staticmethod
    def _postprocess(code: str) -> str:
        # Strip reasoning-model <think>...</think> blocks (and unterminated
        # ones caused by the model running out of tokens mid-thought).
        code = re.sub(r'<think>.*?</think>', '', code, flags=re.DOTALL)
        code = re.sub(r'<think>.*', '', code, flags=re.DOTALL)
        code = code.split('</code>')[0]
        code = code.replace('```python', '')
        code = code.split('```')[0]
        code = code.split('\nEND SOLUTION')[0]
        code = code.replace('<code>', '')
        return code

    def _run_method(self, run_method: Callable, preprocess_method: Callable, 
                   file_path: str, num_workers: int):
        
        with open(file_path, 'w', encoding='utf-8') as f:
            with cfuts.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        run_method,
                        preprocess_method(
                            task) if preprocess_method else task
                    ): task for task in self.dataset
                }

                with tqdm(total=len(futures), desc="Solving Problems") as pbar:
                    for future in cfuts.as_completed(futures):
                        task = futures[future]
                        try:
                            result = self._postprocess(future.result())
                            f.write(json.dumps(
                                {"code": result, "metadata": task.metadata},
                                ensure_ascii=False
                            ) + '\n')
                            f.flush()

                        except Exception as e:
                            print(
                                f"Error processing task {task.metadata}: {str(e)}")
                        finally:
                            pbar.update(1)

    def eval(self, run_method: Callable, preprocess_method: Optional[Callable] = None, 
             save_path: str = 'results', continue_exp: str = None, num_workers: int = 16):
        
        """Evaluates a code generation model on the DS1000 benchmark.

        Args:
            run_method: Function that takes a task (or preprocessed task) and 
                       returns generated code as a string.
            preprocess_method: Optional function to preprocess DataItemDS1000 
                             before passing to run_method. If None, the raw
                             DataItemDS1000 is passed to run_method.
            save_path: Directory path where evaluation results will be saved.
            continue_exp: If provided, continues evaluation from existing experiment
                        directory instead of starting a new one.
            num_workers: Number of parallel workers for code execution.

        Returns:
            ResultsDS1000: Object containing detailed evaluation results including:
                        - Overall accuracy statistics
                        - Per-library performance breakdown  
                        - Per-perturbation type performance
                        - Raw data for custom analysis

        Note:
            Each code sample is executed with a 120-second timeout in isolated
            processes to ensure safety and reproducibility.

        Example:
            >>> # Simple evaluation with custom preprocessing
            >>> def preprocess(task):
            ...     return task.prompt
            >>> 
            >>> def run_model(prompt):
            ...     return "np.ones(5)"  # Example model output
            >>> 
            >>> results = benchmark.eval(
            ...     run_method=run_model,
            ...     preprocess_method=preprocess
            ... )
        """

        if not continue_exp:
            experiment_save_path = os.path.join(
                save_path,
                datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            os.makedirs(experiment_save_path, exist_ok=True)
            file_path = os.path.join(experiment_save_path, 'answers.jsonl')
            self._run_method(run_method, preprocess_method, file_path, num_workers)
        else:
            experiment_save_path = os.path.join(
                save_path,
                continue_exp
            )
            file_path = os.path.join(experiment_save_path, 'answers.jsonl')

        ds1000_results = []
        futs = []

        with cfuts.ProcessPoolExecutor(max_workers=num_workers) as executor:

            with open(file_path, 'r', encoding='utf-8') as f:

                for line in f:
                    answ_dict = json.loads(line)

                    result = answ_dict["code"]
                    problem_id = answ_dict["metadata"]["problem_id"]

                    task = self.dataset[problem_id]

                    test_program = (
                        task.code_context + '\n'
                        + f'code = {repr(result)}\n'
                        + 'test_execution(code)\n'
                        + ('test_string(code)\n' if 'test_string(' in task.code_context else '\n')
                    )

                    futs.append(executor.submit(
                        execution.check_correctness, test_program, timeout=120, metadata=task.metadata))

            for f in tqdm(cfuts.as_completed(futs), total=len(futs), desc="Evaluating"):
                eval_result, metadata = f.result()
                eval_result.update(metadata)
                ds1000_results.append(eval_result)

        ds_1000_results = ResultsDS1000.from_records(data=ds1000_results)
        ds_1000_results.save(experiment_save_path)
