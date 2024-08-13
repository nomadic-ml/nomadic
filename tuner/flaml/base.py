from typing import Any, Dict, Optional
from pydantic import Field

from flaml import tune
from flaml import BlendSearch
from flaml.tune.analysis import ExperimentAnalysis

from nomadic.result import RunResult, TunedResult
from nomadic.tuner import BaseParamTuner


class FlamlParamTuner(BaseParamTuner):
    """
    Parameter tuner powered by FLAML.
    """

    search_alg: Optional[Any] = Field(
        default=BlendSearch(
            metric="score",
            mode="max",
        ),
        description="FLAML search algorithm",
    )
    num_samples: Optional[int] = Field(default=-1, description="FLAML num samples")
    time_budget_s: Optional[int] = Field(
        default=None, description="Time budget (sec) for the experiment to run."
    )
    scheduler: Optional[str] = Field(default=None, description="FLAML scheduler")
    use_ray: Optional[bool] = Field(
        default=False,
        description="Whether to use Ray as FLAML's parameter tuning engine vs. Optuna",
    )

    def model_post_init(self, ctx):
        if self.scheduler not in (
            None,
            "flaml",
            "asha",
            "async_hyperband",
            "asynchyperband",
        ):
            raise NotImplementedError("Given FLAML scheduler option is not supported")

    def fit(self) -> TunedResult:
        def _param_fn_wrapper(param_dict) -> Dict:
            run_result = self.param_fn(param_dict)
            # need to convert RunResult to dict to obey
            # FLAML's API
            tune.report(
                score=run_result.score, config=param_dict, metadata=run_result.metadata
            )
            if self.results_filepath:
                self.add_entries_to_results_json_file(run_result)
            return run_result.model_dump()

        """Run tuning."""
        # Combine fixed, current and search space parameters
        param_dict = self.param_dict if self.param_dict is not None else {}
        fixed_param_dict = (
            self.fixed_param_dict if self.fixed_param_dict is not None else {}
        )
        # current_param_dict = (
        #     self.current_param_dict if self.current_param_dict is not None else {}
        # )

        config = {**param_dict, **fixed_param_dict}
        # config = {**param_dict, **fixed_param_dict, **current_param_dict}

        # Run hyperparameter tuning
        flaml_run_result: ExperimentAnalysis = tune.run(
            _param_fn_wrapper,
            config=config,
            num_samples=self.num_samples,
            time_budget_s=self.time_budget_s,
            scheduler=self.scheduler,
            search_alg=self.search_alg,
            use_ray=self.use_ray,
        )

        # sort the results by score
        run_results = []
        for trial in flaml_run_result.trials:
            run_results.append(
                RunResult(
                    score=trial.last_result[self.search_alg.metric],
                    params=trial.config,
                    metadata=trial.last_result["metadata"],
                )
            )
        return TunedResult(run_results=run_results)
