import gzip
import json
from .data_types import DataItemDS1000


class DatasetDS1000:
    """Dataset loader for the DS1000 benchmark.
    
    Handles reading and parsing the compressed DS1000 dataset file.
    The dataset is expected to be in gzipped JSONL format.
    
    Args:
        path: Path to the .jsonl.gz dataset file.
    
    Example:
        >>> dataset = DatasetDS1000("./data/ds1000.jsonl.gz")
        >>> for item in dataset:
        ...     print(item.prompt)
        ...     break  # Process first item only
        
        >>> item = dataset[0]
    """
    def __init__(self, path):
        self.path = path
        self._items = []
        self._pid_to_item = {}
        
        self._load_data()
    
    def _load_data(self):
        with gzip.open(self.path, "rt") as f:
            for idx, line in enumerate(f):
                # if idx in [0, 999, 498]: # Выкидываем часть примеров из few-shot
                #     continue
                item = self._preprocess(line)
                self._items.append(item)
                self._pid_to_item[item.p_id] = item
    
    def _preprocess(self, line: str) -> DataItemDS1000:
        """Converts a JSON line from the dataset into a DataItemDS1000.
        
        Args:
            line: A single line from the JSONL file as a string.
            
        Returns:
            DataItemDS1000: Parsed and validated data item.
        """
        return DataItemDS1000.from_dict(json.loads(line))
    
    def __iter__(self):
        """Iterates over the dataset yielding DataItemDS1000 objects.
        
        Yields:
            DataItemDS1000: The next data item from the dataset.
        """
        yield from self._items
    
    def __getitem__(self, index) -> DataItemDS1000:
        """Get item by index.
        
        Args:
            index: p_id of DataItemDS1000.
            
        Returns:
            DataItemDS1000: The data item with the specified p_id.
        """

        return self._pid_to_item.get(index, None)