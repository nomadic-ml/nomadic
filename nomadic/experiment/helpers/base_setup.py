from typing import Any, Dict, Callable, Optional, List
from abc import ABC, abstractmethod
from pydantic import BaseModel
import logging



def setup_tuner(self, param_dict: Dict[str, Any], param_function: Callable, **kwargs) -> Any:
    """Set up parameter tuner with default configuration."""
    try:
        # Extract tuning parameters
        tuning_params = {
            'learning_rate': param_dict.get('learning_rate', 0.01),
            'num_iterations': param_dict.get('num_iterations', 100),
            'batch_size': param_dict.get('batch_size', 32),
            'early_stopping': param_dict.get('early_stopping', True)
        }

        # Apply parameter function if provided
        if param_function:
            tuning_params = param_function(tuning_params)

        # Add any additional kwargs
        tuning_params.update(kwargs)

        return tuning_params

    except Exception as e:
        logging.error(f"Error setting up tuner: {str(e)}")
        # Return default parameters as fallback
        return {
            'learning_rate': 0.01,
            'num_iterations': 100,
            'batch_size': 32,
            'early_stopping': True
        }

def enforce_param_types(self, param_values: Dict[str, Any], model_hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce parameter types based on model hyperparameters."""
    try:
        enforced_params = {}

        for param_name, param_value in param_values.items():
            # Get expected type from hyperparameters
            expected_type = model_hyperparameters.get(param_name, {}).get('type', type(param_value))

            # Convert value to expected type
            if expected_type == bool:
                enforced_params[param_name] = bool(param_value)
            elif expected_type == int:
                enforced_params[param_name] = int(float(param_value))
            elif expected_type == float:
                enforced_params[param_name] = float(param_value)
            elif expected_type == str:
                enforced_params[param_name] = str(param_value)
            elif expected_type == list:
                enforced_params[param_name] = list(param_value)
            elif expected_type == dict:
                enforced_params[param_name] = dict(param_value)
            else:
                # Keep original value if type is not recognized
                enforced_params[param_name] = param_value

        return enforced_params

    except Exception as e:
        logging.error(f"Error enforcing parameter types: {str(e)}")
        # Return original parameters as fallback
        return param_values
