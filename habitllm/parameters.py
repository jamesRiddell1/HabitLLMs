import json
from pathlib import Path

CONFIG_PATH = Path("extensions/habitllm/config.json")

class Parameters:
    _instance = None
    
    @staticmethod
    def getInstance():
        if Parameters._instance is None:
            Parameters()
        return Parameters._instance
    
    def __init__(self):
        if Parameters._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Parameters._instance = self
            self.hyperparameters = self._load_file_from_json(CONFIG_PATH)
            
            
    def _load_file_from_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        
        return data

def get_active_routine() -> str:
    return Parameters.getInstance().hyperparameters['routines']['default']

def get_routine_choices() -> str:
    return Parameters.getInstance().hyperparameters['routines']['categories']

def get_is_inference_specific_context() -> bool:
    return Parameters.getInstance().hyperparameters['inference specific context']['default']

def get_inference_specific_context_chunks() -> int:
    return Parameters.getInstance().hyperparameters['inference specific context chunks']['default']

def get_is_model_persistent_context() -> bool:
    return Parameters.getInstance().hyperparameters['model persistent context']['default']

def get_model_persistent_context_chunks() -> int:
    return Parameters.getInstance().hyperparameters['model persistent context chunks']['default']

def set_active_routine(value: str):
    Parameters.getInstance().hyperparameters['routines']['default'] = value

def set_is_inference_specific_context(value: bool):
    Parameters.getInstance().hyperparameters['inference specific context']['default'] = value

def set_is_model_persistent_context(value: bool):
    Parameters.getInstance().hyperparameters['model persistent context']['default'] = value
    
def set_inference_specific_context_chunks(value: int):
    Parameters.getInstance().hyperparameters['inference specific context chunks']['default'] = value

def set_model_persistent_context_chunks(value: int):
    Parameters.getInstance().hyperparameters['model persistent context chunks']['default'] = value
            
            