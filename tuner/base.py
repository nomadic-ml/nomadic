from abc import abstractmethod
import itertools
import json
import os
from typing import Any, Callable, Dict, List, Optional
from matplotlib import pyplot as plt
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from ray import tune as ray_tune
from ray.train import RunConfig

from nomadic.result import RunResult, TunedResult
from nomadic.util import get_tqdm_iterable


class BaseParamTuner(BaseModel):
    """Base param tuner."""

    param_fn: Callable[[Dict[str, Any]], RunResult] = Field(
        ..., description="Function to run with parameters."
    )
    param_dict: Dict[str, Dict[str, Any]] = Field(
        ..., description="A dictionary of parameters to iterate over."
    )
    fixed_param_dict: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="A dictionary of fixed parameters passed to each job.",
    )
    current_param_dict: Optional[Dict[str, Any]] = Field(
        default=None,
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
    def fit(self) -> TunedResult:
        """Tune parameters."""

    def add_entries_to_results_json_file(self, new_entry: RunResult) -> None:
        if not self.results_filepath:
            return
        try:
            # Read the existing JSON file
            with open(self.results_filepath, "r") as file:
                print(f"existing file: {self.results_filepath}")
                data = json.load(file)
                print(f"existing data: {data}")
        except (FileNotFoundError, json.JSONDecodeError):
            # If the file does not exist or is empty/invalid, initialize an empty list
            data = []

        # Ensure data is a list before updating it
        if not isinstance(data, list):
            data = []

        print(f"new data: {new_entry.model_dump()}")
        # Append the new entries to the list
        data.extend([new_entry.model_dump()])

        # Write the updated list back to the JSON file
        with open(self.results_filepath, "w") as file:
            print(f"updating file: {self.results_filepath}")
            json.dump(data, file, indent=4)
            print(f"updated data: {data}")

    def save_results_table(self, results, filepath):
        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False)

    def save_graphs(self, results, output_dir):
        # Add a new column for the JSON string of hyperparameters
        results["hyperparameters"] = results.apply(
            lambda x: json.dumps(
                {
                    "branching_factor": x["param_branching_factor"],
                    "width": x["param_width"],
                    "depth": x["param_depth"],
                },
                sort_keys=True,
            ),
            axis=1,
        )

        # Calculate scores based on the 'jailbroken' column
        results["score"] = results["jailbroken"].apply(lambda x: 10 if x else 1)

        # Group by hyperparameters and calculate means
        grouped = (
            results.groupby("hyperparameters")
            .agg(
                {
                    "score": "mean",
                    "attack_api_calls": "mean",
                    "target_api_calls": "mean",
                    "judge_api_calls": "mean",
                    "total_cost": "mean",
                }
            )
            .reset_index()
        )

        # Plotting: Score vs. Hyperparameters
        plt.figure(figsize=(12, 6))
        plt.bar(grouped["hyperparameters"], grouped["score"], color="blue")
        plt.xlabel("Hyperparameters")
        plt.ylabel("Average Score")
        plt.title("Average Score by Hyperparameters")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "average_scores_by_hyperparameters.png"))
        plt.clf()

        # Plotting: Total API Calls vs. Hyperparameters (aggregated from all APIs)
        grouped["total_api_calls"] = (
            grouped["attack_api_calls"]
            + grouped["target_api_calls"]
            + grouped["judge_api_calls"]
        )
        plt.figure(figsize=(12, 6))
        plt.bar(grouped["hyperparameters"], grouped["total_api_calls"], color="green")
        plt.xlabel("Hyperparameters")
        plt.ylabel("Average Total API Calls")
        plt.title("Average Total API Calls by Hyperparameters")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "average_total_api_calls_by_hyperparameters.png")
        )
        plt.clf()

        # Plotting: Total Cost vs. Hyperparameters
        plt.figure(figsize=(12, 6))
        plt.bar(grouped["hyperparameters"], grouped["total_cost"], color="red")
        plt.xlabel("Hyperparameters")
        plt.ylabel("Average Total Cost")
        plt.title("Average Total Cost by Hyperparameters")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "average_total_cost_by_hyperparameters.png")
        )
        plt.clf()

        print("Graphs have been saved successfully.")


class RayTuneParamTuner(BaseParamTuner):
    """
    Parameter tuner powered by Ray Tune.
    Example:
        >>> from ray import tune
        >>> param_space={
        ...   "x": tune.grid_search([10, 20]),
        ...   "y": tune.grid_search(["a", "b", "c"])
        ... }

    Args:
        param_dict(Dict): A dictionary of parameters to iterate over.
            Example param_dict:
            {
                "num_epochs": [10, 20],
                "batch_size": [8, 16, 32],
            }
        fixed_param_dict(Dict): A dictionary of fixed parameters passed to each job.

    """

    run_config_dict: Optional[dict] = Field(
        default=None, description="Run config dict for Ray Tune."
    )
    search_method: Optional[str] = Field(
        default="grid", description="Ray Tune search option. Can be: [grid, bayesian]"
    )

    def model_post_init(self, ctx):
        if self.search_method not in ("grid", "bayesian"):
            raise ValueError(
                f"Selected RayTuneParamTuner search_method `{self.search_method}` is not valid."
            )

    def _param_fn_wrapper(
        self,
        ray_param_dict: Dict,
        fixed_param_dict: Optional[Dict] = None,
    ) -> Dict:
        # need a wrapper to pass in parameters to ray_tune + fixed params
        fixed_param_dict = fixed_param_dict or {}
        full_param_dict = {
            **fixed_param_dict,
            **ray_param_dict,
        }
        tuned_result = self.param_fn(full_param_dict)
        # need to convert RunResult to dict to obey
        # Ray Tune's API
        return tuned_result.model_dump()

    def _convert_ray_tune_run_result(
        self, result_grid: ray_tune.ResultGrid
    ) -> RunResult:
        # convert dict back to RunResult (reconstruct it with metadata)
        # get the keys in RunResult, assign corresponding values in
        # result.metrics to those keys
        try:
            run_result = RunResult.model_validate(result_grid.metrics)
        except ValidationError:
            # Tuning function may have errored out (e.g. due to objective function erroring)
            # Handle gracefully
            run_result = RunResult(score=-1, params={})

        # add some more metadata to run_result (e.g. timestamp)
        run_result.metadata["timestamp"] = (
            result_grid.metrics["timestamp"] if result_grid.metrics else None
        )
        return run_result

    def _set_ray_tuner(self, param_space, search_method=None):
        search_method = (
            search_method if search_method is not None else self.search_method
        )
        run_config = RunConfig(**self.run_config_dict) if self.run_config_dict else None
        if search_method == "grid":
            return ray_tune.Tuner(
                ray_tune.with_parameters(
                    self._param_fn_wrapper, fixed_param_dict=self.fixed_param_dict
                ),
                param_space=param_space,
                run_config=run_config,
            )
        elif search_method == "bayesian":
            from ray.tune.search.bayesopt import BayesOptSearch

            new_param_space = {
                hp_name: ray_tune.uniform(*tuple(val["uniform"]))
                for hp_name, val in param_space.items()
            }
            return ray_tune.Tuner(
                ray_tune.with_parameters(
                    self._param_fn_wrapper, fixed_param_dict=self.fixed_param_dict
                ),
                # TODO: Generalize metric name, mode, num_samples
                tune_config=ray_tune.TuneConfig(
                    search_alg=BayesOptSearch(metric="score", mode="max"), num_samples=8
                ),
                param_space=new_param_space,
                run_config=run_config,
            )
        else:
            raise NotImplementedError

    def fit(self) -> TunedResult:
        """Run tuning."""
        ray_param_dict = self.param_dict
        tuner = self._set_ray_tuner(ray_param_dict)
        result_grids = tuner.fit()
        all_run_results = [
            self._convert_ray_tune_run_result(result_grid)
            for result_grid in result_grids
        ]

        # If current_hp_values is specified, ensure current hp values are also scored,
        # and added to results.
        is_current_hps_in_results = False
        if self.current_param_dict:
            for run_result in all_run_results:
                # Current hp values have already been tested, and their evaluation has already been complete.
                # There is no need to re-run the evaluation of current HP values.
                if all(
                    item in run_result.params.items()
                    for item in self.current_param_dict.items()
                ):
                    is_current_hps_in_results = True
                    break
            if not is_current_hps_in_results:
                all_run_results.append(
                    self._convert_ray_tune_run_result(
                        self._set_ray_tuner(
                            param_space={
                                hp_name: ray_tune.grid_search([val])
                                for hp_name, val in self.current_param_dict.items()
                            },
                            search_method="grid",
                        ).fit()[0]
                    )
                )

        # sort the results by score
        sorted_run_results = sorted(
            all_run_results, key=lambda x: x.score, reverse=True
        )

        return TunedResult(run_results=sorted_run_results, best_idx=0)


# TODO: Finish implementing ParamTuner
class ParamTuner(BaseParamTuner):
    def fit(self) -> TunedResult:
        def generate_param_combinations(
            param_dict: Dict[str, Any]
        ) -> List[Dict[str, Any]]:
            """Generate parameter combinations."""
            param_names = []
            param_values = []

            for param_name, param_val in param_dict.items():
                if "grid_search" in param_val:
                    param_names.append(param_name)
                    param_values.append(param_val["grid_search"])

            combinations = [
                dict(zip(param_names, combination))
                for combination in itertools.product(*param_values)
            ]
            return combinations

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
                run_result = self.param_fn(full_param_dict)
                all_run_results.append(run_result)

        # sort the results by score
        sorted_run_results = sorted(
            all_run_results, key=lambda x: x.score, reverse=True
        )

        return TunedResult(run_results=sorted_run_results, best_idx=0)
