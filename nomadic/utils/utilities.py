"""Utility functions for experiment management."""
import ast
import time
from typing import TypeVar, Callable, Any, List
from functools import wraps

T = TypeVar('T')


def convert_string_to_list(s: str) -> List[Any]:
    try:
        # Use literal_eval to safely evaluate the string to a Python object
        result = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        raise ValueError("Invalid string representation of a list.")
    if isinstance(result, list):
        try:
            # Check if all elements are strings that can be converted to floats
            if all(
                isinstance(x, str) and x.replace(".", "", 1).isdigit()
                for x in result
            ):
                return [float(x) for x in result]
            # Check if all elements are already floats or ints
            elif all(isinstance(x, float) for x in result):
                return [float(x) for x in result]
            elif all(isinstance(x, int) for x in result):
                return [int(x) for x in result]
            # Otherwise, return the list as is (assuming it's a list of strings)
            else:
                return result
        except (ValueError, SyntaxError):
            raise ValueError("Invalid string representation of a list.")
    else:
        raise ValueError("The provided string is not a list.")


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 60.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that implements retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retries before giving up
        initial_delay: Initial delay between retries in seconds
        exponential_base: Base for exponential backoff
        max_delay: Maximum delay between retries in seconds

    Returns:
        Decorator function that implements retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception = None

            for retry in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if retry == max_retries:
                        raise last_exception

                    time.sleep(min(delay, max_delay))
                    delay *= exponential_base

            raise last_exception  # type: ignore
        return wrapper
    return decorator

def transform_eval_dataset_to_eval_json(eval_dataset):
    eval_json = {
        "queries": {},
        "responses": {}
    }
    # Loop through each entry in eval_dataset
    for idx, entry in enumerate(eval_dataset, start=1):
        query_key = f"query{idx}"
        eval_json["queries"][query_key] = entry['query']
        eval_json["responses"][query_key] = entry['answer']
    return eval_json
