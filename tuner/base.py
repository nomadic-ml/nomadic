from abc import abstractmethod
import itertools
import random
from typing import Any, Callable, Dict, Iterable, List, Optional
from pydantic import BaseModel, Field, ValidationError

from nomadic.result import RunResult, TunedResult


class BaseParamTuner(BaseModel):
    """Base param tuner."""

    param_fn: Callable[[Dict[str, Any]], RunResult] = Field(
        ..., description="Function to run with parameters."
    )
    param_dict: Dict[str, Any] = Field(
        ..., description="A dictionary of parameters to iterate over."
    )
    fixed_param_dict: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary of fixed parameters passed to each job.",
    )
    current_param_dict: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional dictionary of current hyperparameter values.",
    )
    show_progress: bool = False

    @abstractmethod
    def fit(self) -> TunedResult:
        """Tune parameters."""


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

    def fit(self) -> TunedResult:
        """Run tuning."""
        from ray import tune as ray_tune
        from ray.train import RunConfig
        from ray.tune.result_grid import ResultGrid

        ray_param_dict = self.param_dict

        def param_fn_wrapper(
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

        def convert_ray_tune_run_result(result_grid: ResultGrid) -> RunResult:
            # convert dict back to RunResult (reconstruct it with metadata)
            # get the keys in RunResult, assign corresponding values in
            # result.metrics to those keys
            try:
                run_result = RunResult.model_validate(result_grid.metrics)
            except ValidationError as e:
                # Tuning function may have errored out (e.g. due to objective function erroring)
                # Handle gracefully
                run_result = RunResult(score=-1, params={})

            # add some more metadata to run_result (e.g. timestamp)
            run_result.metadata["timestamp"] = (
                result_grid.metrics["timestamp"]
                if result_grid.metrics
                else None
            )
            return run_result

        run_config = (
            RunConfig(**self.run_config_dict) if self.run_config_dict else None
        )
        tuner = ray_tune.Tuner(
            ray_tune.with_parameters(
                param_fn_wrapper, fixed_param_dict=self.fixed_param_dict
            ),
            param_space=ray_param_dict,
            run_config=run_config,
        )

        result_grids = tuner.fit()

        all_run_results = [
            convert_ray_tune_run_result(result_grid)
            for result_grid in result_grids
        ]

        # If current_hp_values is specified, ensure current hp values are also scored,
        # and added to results.
        is_current_hps_in_results = False
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
                convert_ray_tune_run_result(
                    ray_tune.Tuner(
                        ray_tune.with_parameters(
                            param_fn_wrapper,
                            fixed_param_dict=self.fixed_param_dict,
                        ),
                        param_space={
                            hp_name: ray_tune.grid_search([val])
                            for hp_name, val in self.current_param_dict.items()
                        },
                        run_config=run_config,
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
    search_method: str = Field(
        ..., description="Search method: 'grid' or 'random'."
    )
    n_iter: int = Field(
        default=10, description="Number of iterations for random search."
    )

    def fit(self) -> TunedResult:
        if self.search_method not in ["grid", "random"]:
            raise ValueError("search_method must be either 'grid' or 'random'.")

        run_results = []

        if self.search_method == "grid":
            param_combinations = list(
                itertools.product(*self.param_dict.values())
            )
            for params in param_combinations:
                param_set = dict(zip(self.param_dict.keys(), params))
                param_set.update(self.fixed_param_dict)
                score, metadata = self.evaluate(param_set)
                run_results.append(
                    RunResult(score=score, params=param_set, metadata=metadata)
                )

        elif self.search_method == "random":
            for _ in range(self.n_iter):
                param_set = {
                    k: random.choice(v) for k, v in self.param_dict.items()
                }
                param_set.update(self.fixed_param_dict)
                score, metadata = self.evaluate(param_set)
                run_results.append(
                    RunResult(score=score, params=param_set, metadata=metadata)
                )

        best_idx = max(
            range(len(run_results)), key=lambda i: run_results[i].score
        )
        return TunedResult(run_results=run_results, best_idx=best_idx)
