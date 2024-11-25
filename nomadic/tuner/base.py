from abc import abstractmethod
import itertools
import json
from typing import Any, Callable, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field

try:
    import ray.tune.search.sample as my_sample
except (ImportError, AssertionError):
    import flaml.tune.sample as my_sample

from nomadic.result import RunResult, ExperimentResult
from nomadic.util import get_tqdm_iterable


class BaseParamTuner(BaseModel):
    """Base param tuner."""

    param_fn: Callable[[Dict[str, Any]], Any] = Field(
        ..., description="Function to run with parameters."
    )
    param_dict: Optional[Dict[str, Any]] = Field(
        default=None, description="A dictionary of parameters to iterate over."
    )
    fixed_param_dict: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="A dictionary of fixed parameters passed to each job.",
    )
    current_param_dict: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional dictionary of current hyperparameter values.",
    )
    show_progress: bool = False
    num_prompts: int = Field(
        default=1,
        description="Number of prompt variations to generate for each data point.",
    )
    results_filepath: Optional[str] = Field(
        default=None, description="Path of outputting tuner run results."
    )

    @abstractmethod
    def fit(self) -> ExperimentResult:
        """Tune parameters."""

    def add_entries_to_results_json_file(self, new_entry: RunResult) -> None:
        if not self.results_filepath:
            return
        try:
            # Read the existing JSON file
            with open(self.results_filepath, "r") as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            # If the file does not exist or is empty/invalid, initialize an empty list
            data = []

        # Ensure data is a list before updating it
        if not isinstance(data, list):
            data = []

        # Get the model dump of new_entry
        entry_dict = new_entry.model_dump()

        # Remove the 'visualization' field if it exists, as bytes
        # it can't be dumped to JSON.
        if 'visualization' in entry_dict:
            del entry_dict['visualization']

        # Append the new entry to the list
        data.append(entry_dict)

        # Write the updated list back to the JSON file
        with open(self.results_filepath, "w") as file:
            json.dump(data, file, indent=4)


    def save_results_table(self, results, filepath):
        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False)


class ParamTuner(BaseParamTuner):
    def fit(self) -> ExperimentResult:
        def generate_param_combinations(
            search_space: Dict[str, Any]
        ) -> List[Dict[str, Any]]:
            # Helper function to get possible values for each hyperparameter
            def get_values(param):
                if isinstance(param, my_sample.Categorical):
                    return param.categories
                elif isinstance(param, my_sample.Integer):
                    return list(range(param.lower, param.upper))
                else:
                    raise ValueError(f"Unsupported parameter type: {type(param)}")

            # Extract hyperparameter names and their respective ranges
            keys = search_space.keys()
            values = [get_values(search_space[k]) for k in keys]

            # Generate all possible combinations
            return [dict(zip(keys, v)) for v in itertools.product(*values)]

        param_combinations = generate_param_combinations(self.param_dict)

        # for each combination, run the job with the arguments
        # in args_dict

        combos_with_progress = enumerate(
            get_tqdm_iterable(
                param_combinations, self.show_progress, "Param combinations."
            )
        )

        all_run_results = []
        fixed_param_dict = self.fixed_param_dict or {}
        for idx, param_combination in combos_with_progress:
            full_param_dict = {
                **fixed_param_dict,
                **param_combination,
            }
            for _ in range(self.num_prompts):
                result = self.param_fn(full_param_dict)
                all_run_results.append(
                    RunResult(score=result, params=full_param_dict, metadata={})
                )

        return ExperimentResult(run_results=all_run_results)
