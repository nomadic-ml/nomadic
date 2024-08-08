from typing import Any, Optional
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
    scheduler: Optional[str] = Field(default=None, description="FLAML scheduler")

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
        """Run tuning."""
        # Combine fixed, current and search space parameters
        self.param_dict = self.param_dict if not None else {}
        self.current_param_dict = self.current_param_dict if not None else {}
        self.fixed_param_dict = self.fixed_param_dict if not None else {}
        config = {**self.param_dict, **self.current_param_dict, **self.fixed_param_dict}

        # Run hyperparameter tuning
        flaml_run_result: ExperimentAnalysis = tune.run(
            self.param_fn,
            config=config,
            num_samples=self.num_samples,
            scheduler=self.scheduler,
            search_alg=self.search_alg,
        )

        # sort the results by score
        run_results = []
        for trial in flaml_run_result.trials:
            run_results.append(
                RunResult(
                    score=trial.last_result[self.search_alg.metric],
                    params=trial.config,
                    metadata=trial.metric_analysis,
                )
            )
        sorted_run_results = sorted(run_results, key=lambda x: x.score, reverse=True)

        return TunedResult(run_results=sorted_run_results, best_idx=0)
