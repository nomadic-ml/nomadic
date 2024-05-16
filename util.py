from typing import Iterable

def get_tqdm_iterable(items: Iterable, show_progress: bool, desc: str) -> Iterable:
    """
    Implement tqdm progress bar.
    """
    _iterator = items
    if show_progress:
        try:
            from tqdm.auto import tqdm
            return tqdm(items, desc=desc)
        except ImportError:
            pass
    return _iterator

import ast
import subprocess
import time
import pickle
from pathlib import Path

from llama_index.experimental.param_tuner.base import TunedResult

def convert_string_to_int_array(string: str) -> list:
    try:
        int_array = ast.literal_eval(string)
        if not isinstance(int_array, list) or not all(isinstance(i, int) for i in int_array):
            raise ValueError("The input string does not represent an integer array.")
        return int_array
    except (ValueError, SyntaxError) as e:
        raise ValueError("Invalid input string") from e

def execute_bash_command(command: str) -> str:
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e.stderr}"

def save_tuned_result_to_local(tuned_result: TunedResult, sample_name, evals_folder_path):
    current_time = int(time.time())
    path = Path(f"{evals_folder_path}/{sample_name}")
    path.mkdir(parents=True, exist_ok=True)
    with open(f"{path}/TunedResult_{current_time}.pkl", 'wb') as picklefile:
        pickle.dump(tuned_result, picklefile)

def read_serialized_tuned_result(pickled_file_path: str) -> TunedResult:
    with open(pickled_file_path, 'rb') as pickled_file:
        file_contents = pickle.load(pickled_file)
    return file_contents