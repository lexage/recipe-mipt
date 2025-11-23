from typing import Dict, Type, Optional, Any, List
import inspect
from src.pipelines.configs import ComponentConfig
from src.pipelines.constants import ComponentNames
from src.agent_constructor.core import Block

COMPONENT_REGISTRY: Dict[ComponentNames, Dict] = {}


def register_component(
    component_type: ComponentNames,
):
    """
    Декоратор для автоматической регистрации компонента.
    
    Args:
        component_type: Строковый идентификатор типа компонента
    
    Usage:
        @register_component(ComponentNames.MY_NEW_AGENT)
        class MyNewAgent(Agent):
            def __init__(self, name: str, url: str, model_name: str):
                ...
    """
    
    def wrapper(component_class: Type):

        sig = inspect.signature(component_class.__init__)
        
        dependencies = [
            param.name 
            for param in sig.parameters.values()
            if param.name != 'self'
            and param.annotation != inspect.Parameter.empty
            and inspect.isclass(param.annotation)
            and issubclass(param.annotation, Block)
        ]
    
        COMPONENT_REGISTRY[component_type] = {
            'class': component_class,
            'dependencies': dependencies
        }
                
        return component_class
    
    return wrapper

def get_component_dependencies(component_config: ComponentConfig):
    component_info = COMPONENT_REGISTRY.get(component_config.type, {})
    if component_info:
        return component_info.get('dependencies')
    else:
        raise ValueError(f"Unregistred component: {component_config.type}")

def create_component_automatically(
    config: ComponentConfig,
    dependencies: Dict[str, Any],
    component_class: Type,
    dependencies_names: List[str]
) -> Any:
    """
    Автоматически создает компонент, анализируя сигнатуру конструктора.
    
    Параметры берутся из:
    1. config.params - для обычных параметров
    2. dependencies - для зависимостей (с учетом dependency_mapping)
    """
    # Получаем сигнатуру конструктора
    sig = inspect.signature(component_class.__init__)
    params = {}
    
    # Пропускаем 'self'
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
            
        if param_name in dependencies_names:
            params[param_name] = dependencies[param_name]

        elif param_name in config.params:
            # Берем из конфига
            params[param_name] = config.params[param_name]
        elif param.default != inspect.Parameter.empty:
            # Используем значение по умолчанию
            continue
        else:
            raise ValueError(
                f"Required parameter '{param_name}' not found for {component_class.__name__}. "
                f"Available: params={list(config.params.keys())}, "
                f"dependencies={list(dependencies.keys())}"
            )
    
    return component_class(**params)
