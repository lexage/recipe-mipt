import pandas as pd
import os
from dataclasses import dataclass


@dataclass
class DataItemDS1000:
    """Represents a single data item from the DS1000 benchmark dataset.
    
    Attributes:
        prompt: The natural language prompt describing the coding task.
        reference_code: The ground truth/reference implementation code.
        metadata: Additional metadata about the task (library, perturbation type, etc.).
        code_context: The code context including imports and test functions.
    
    Example:
        >>> data = {
        ...     'prompt': 'Create a numpy array of ones',
        ...     'reference_code': 'np.ones(5)',
        ...     'metadata': {'library': 'numpy', 'perturbation_type': 'origin'},
        ...     'code_context': 'import numpy as np\\ndef test_execution(code): ...'
        ... }
        >>> item = DataItemDS1000.from_dict(data)
        >>> print(item.prompt)
        'Create a numpy array of ones'
    """

    p_id: str
    prompt: str
    reference_code: str
    metadata: dict
    code_context: str

    @classmethod
    def from_dict(cls, data: dict) -> 'DataItemDS1000':
        """Creates a DataItemDS1000 instance from a dictionary.
        
        Args:
            data: Dictionary containing the item data with keys:
                - 'prompt': The task prompt
                - 'reference_code': Reference implementation
                - 'metadata': Task metadata
                - 'code_context': Code context with imports and tests
        
        Returns:
            DataItemDS1000: Initialized data item instance.
        """

        return cls(
            p_id=data['metadata']['problem_id'],
            prompt=data['prompt'],
            reference_code=data['reference_code'],
            metadata=data['metadata'],
            code_context=data['code_context'],
        )


@dataclass
class ResultsDS1000:
    """Container for DS1000 benchmark evaluation results.
    
    Provides methods for analyzing, summarizing and saving benchmark results.
    
    Attributes:
        df: DataFrame containing detailed evaluation results with columns:
            - passed: bool indicating if test passed
            - score: 1 if passed, 0 otherwise  
            - library: which library the test was for (numpy, pandas, etc.)
            - perturbation_type: type of code perturbation
            - Additional metadata from the evaluation
    
    Example:
        >>> results = benchmark.eval(run_method=my_model)
        >>> print(results.summary())
        >>> results.save("./results/")
    """
    df: pd.DataFrame

    @classmethod
    def from_records(cls, data: list) -> 'ResultsDS1000':
        """Creates ResultsDS1000 from a list of evaluation records.
        
        Args:
            data: List of dictionaries containing evaluation results.
                  Each dict should contain 'score', 'library', 'perturbation_type',
                  and other evaluation metrics.
        
        Returns:
            ResultsDS1000: Initialized results container.
        """
        return cls(df=pd.DataFrame.from_records(data))
    
    def summary(self) -> str:
        """Generates a comprehensive summary of evaluation results.
        
        Returns:
            str: Formatted summary string containing:
                - Overall score statistics (count, mean)
                - Score breakdown by library
                - Score breakdown by perturbation type
        """
        pd.set_option('display.precision', 3)
        summary = self.df.agg({'score': ['count', 'mean']}).to_string()
        summary += '\n' + self.df[['library', 'score']].groupby('library').agg({'score': ['count', 'mean']}).to_string()
        summary += '\n' + self.df[['perturbation_type', 'score']].groupby('perturbation_type').agg({'score': ['count', 'mean']}).to_string()
        return summary
    
    def save(self, path) -> str:
        """Saves results to timestamped directory.
        
        Args:
            base_path: Base directory where results will be saved.
                     A timestamped subdirectory will be created.
        
        Returns:
            str: Path to the created directory, or None if saving failed.
        
        Example:
            >>> results.save("./experiments/")
            './experiments/20231201_143022'
        """
        
        try:
            os.makedirs(path, exist_ok=True)
            
            with open(os.path.join(path, 'summary.txt'), 'w', encoding='utf-8') as f:
                f.write(self.summary())
            
            self.df.to_csv(os.path.join(path, 'results.csv'), index=False)
            
            return path
        
        except Exception as e:
            print(f"Ошибка при сохранении результатов: {e}")
            return None
