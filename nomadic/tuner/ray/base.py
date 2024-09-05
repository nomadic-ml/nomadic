from typing import Dict, Optional
from pydantic import Field, ValidationError

# Assumes ray is installed.
# File should not be imported otherwise.
from ray import tune as ray_tune
from ray.train import RunConfig

from nomadic.result import RunResult, ExperimentResult
from nomadic.tuner import BaseParamTuner


class RayTuneParamTuner(BaseParamTuner):
    """
    Parameter tuner powered by Ray Tune.

    Args:
        param_dict(Dict[str, Any]): A dictionary of parameters to iterate over.
            Example param_dict:
            {
                "num_epochs": tune.randint(10, 20),
                "batch_size": tune.choice([8, 16, 32]),
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
        fixed_param_dict = fixed_param_dict if fixed_param_dict is not None else {}
        full_param_dict = {
            **fixed_param_dict,
            **ray_param_dict,
        }
        experiment_result = self.param_fn(full_param_dict)
        # need to convert RunResult to dict to obey
        # Ray Tune's API
        return experiment_result.model_dump()

    def _convert_ray_tune_run_result(
        self, result_grid: ray_tune.ResultGrid
    ) -> RunResult:
        # convert dict back to RunResult (reconstruct it with metadata)
        # get the keys in RunResult, assign corresponding values in
        # result.metrics to those keys
        try:
            run_result = RunResult.model_validate(result_grid.metrics)
        except ValidationError as e:
            # Tuning function may have errored out (e.g. due to objective function erroring)
            # Handle gracefully
            run_result = RunResult(score=-1, params={}, metadata={"error": e.stderr})

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
            new_param_space = {
                hp_name: ray_tune.grid_search(val.categories)
                for hp_name, val in param_space.items()
                if hasattr(val, "categories")
            }
            ray_tuner = ray_tune.Tuner(
                ray_tune.with_parameters(
                    self._param_fn_wrapper, fixed_param_dict=self.fixed_param_dict
                ),
                param_space=new_param_space,
                run_config=run_config,
            )
        elif search_method == "bayesian":
            from ray.tune.search.bayesopt import BayesOptSearch

            new_param_space = {
                hp_name: ray_tune.uniform(*tuple(val["uniform"]))
                for hp_name, val in param_space.items()
                if hasattr(val, "uniform")
            }
            ray_tuner = ray_tune.Tuner(
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
        return ray_tuner

    def fit(self) -> ExperimentResult:
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

        return ExperimentResult(
            hp_search_space=self.param_dict, run_results=all_run_results
        )
