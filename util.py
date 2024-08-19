from typing import Iterable, List, Type

import ast
import subprocess
import time
import pickle
from pathlib import Path

from nomadic.result.base import ExperimentResult
from pydantic import BaseModel


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


def convert_string_to_int_array(string: str) -> list:
    try:
        int_array = ast.literal_eval(string)
        if not isinstance(int_array, list) or not all(
            isinstance(i, int) for i in int_array
        ):
            raise ValueError("The input string does not represent an integer array.")
        return int_array
    except (ValueError, SyntaxError) as e:
        raise ValueError("Invalid input string") from e


def execute_bash_command(command: str) -> str:
    try:
        result = subprocess.run(
            command, shell=True, check=True, text=True, capture_output=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e.stderr}"


def save_experiment_result_to_local(
    experiment_result: ExperimentResult, sample_name, evals_folder_path
):
    current_time = int(time.time())
    path = Path(f"{evals_folder_path}/{sample_name}")
    path.mkdir(parents=True, exist_ok=True)
    with open(f"{path}/TunedResult_{current_time}.pkl", "wb") as picklefile:
        pickle.dump(experiment_result, picklefile)


def read_serialized_tuned_result(pickled_file_path: str) -> ExperimentResult:
    with open(pickled_file_path, "rb") as pickled_file:
        file_contents = pickle.load(pickled_file)
    return file_contents


def get_subclasses(cls: Type[BaseModel]) -> List[Type[BaseModel]]:
    """Get a list of all subclasses of a Pydantic model."""

    subclasses = []
    for subclass in cls.__subclasses__():
        subclasses.extend(get_subclasses(subclass))
    return subclasses


def is_ray_installed() -> bool:
    """Verify the Ray Tune framework is installed"""
    try:
        from ray import __version__ as ray_version

        assert ray_version >= "1.10.0"
    except (ImportError, AssertionError):
        return False
    else:
        return True
