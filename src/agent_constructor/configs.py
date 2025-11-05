from typing import Optional, List, Union, Dict, Any
from pydantic import BaseModel, Field

from typing import Union, Type
from pydantic import BaseModel

# Базовый класс для всех конфигов
class ComponentConfig(BaseModel):
    type: str
    params: Dict[str, Any] = {}
    dependencies: List[str] = []

# Специфические конфиги наследуются от базового
class DBConfig(ComponentConfig):
    pass

class AgentConfig(ComponentConfig):
    pass

class RetrieverConfig(ComponentConfig):
    pass

class FilterConfig(ComponentConfig):
    pass

class ChunkerConfig(ComponentConfig):
    pass

# Тип для аннотаций
AnyConfig = Union[DBConfig, AgentConfig, RetrieverConfig, FilterConfig, ChunkerConfig]

# Основная конфигурация пайплайна
class PipelineConfig(BaseModel):
    execution_order: List[str] = Field(..., description="Порядок инициализации компонентов")
    components: Dict[str, AnyConfig]
