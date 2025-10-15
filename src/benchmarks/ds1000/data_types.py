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
            prompt=data['prompt'],
            reference_code=data['reference_code'],
            metadata=data['metadata'],
            code_context=data['code_context'],
        )
