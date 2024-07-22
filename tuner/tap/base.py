import itertools
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from skopt import gp_minimize
from skopt.space import Integer
from flaml import tune
from flaml import BlendSearch
from nomadic.result import RunResult, TunedResult
from nomadic.tuner import BaseParamTuner
from nomadic.util import get_tqdm_iterable


class BaseEvaluator(BaseModel):
    goal: str = Field(..., description="The goal of the evaluation")

    class Config:
        arbitrary_types_allowed = True


class BaseTarget(BaseModel):
    model: str = Field(..., description="The model to generate responses")

    class Config:
        arbitrary_types_allowed = True


class BaseAttackLLM(BaseModel):
    model: str = Field(..., description="The model to generate attack messages")

    class Config:
        arbitrary_types_allowed = True


DEFAULT_HYPERPARAMETER_SEARCH_SPACE: Dict[str, Any] = {
    "branching_factor": {"type": int, "values": [3, 4]},
    "width": {"type": int, "values": [5, 6]},
    "depth": {"type": int, "values": [5, 6]},
    "temperature": {"type": int, "values": [0.7]},
    "top_p": {"type": int, "values": [1]},
}


class TAPParamTuner(BaseParamTuner):
    """Parameter tuner for TAP hyperparameters."""

    param_fn: Callable[[Dict[str, Any]], Dict[str, Any]] = Field(
        default=None, description="Function to run with parameters."
    )
    param_dict: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: DEFAULT_HYPERPARAMETER_SEARCH_SPACE,
        description="A dictionary of parameters to iterate over.",
    )
    fixed_param_dict: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="A dictionary of fixed parameters passed to each job.",
    )
    dataset: List[Dict[str, Any]] = Field(
        ..., description="Dataset to use for evaluation."
    )
    evaluation_method: int = Field(default=3, description="Evaluation method to use.")
    run_config_dict: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional run configuration dictionary."
    )
    search_method: str = Field(
        default="grid", description="Search method to use for hyperparameter tuning."
    )
    enhanced_function: Optional[Callable] = Field(
        default=None, description="Enhanced function for FLAML optimization."
    )
    num_simulations: int = Field(
        default=10,
        description="Number of simulations to run for each hyperparameter combination.",
    )
    evaluator: Optional[Any] = Field(..., description="Evaluator instance")
    target: Optional[Any] = Field(..., description="Target instance")
    attack_llm: Optional[Any] = Field(..., description="Attack LLM instance")

    def fit(self) -> TunedResult:
        if self.search_method == "grid":
            return self._grid_search()
        elif self.search_method == "bayesian":
            return self._bayesian_optimization()
        elif self.search_method == "flaml":
            return self._flaml_optimization()
        else:
            raise ValueError(f"Unknown search method: {self.search_method}")

    def _grid_search(self) -> TunedResult:
        param_keys, param_values = zip(*self.param_dict.items())
        param_combinations = list(itertools.product(*param_values))

        results = []
        best_score = float("-inf")
        best_params = None

        for params in get_tqdm_iterable(param_combinations, self.show_progress):
            param_dict = dict(zip(param_keys, params))
            score, result = self._evaluate_params(param_dict)

            results.append(result)

            if score > best_score:
                best_score = score
                best_params = param_dict

        return TunedResult(
            best_params=best_params,
            best_score=best_score,
            results=results,
        )

    def _bayesian_optimization(self) -> TunedResult:
        def objective(params):
            param_dict = dict(zip(self.param_dict.keys(), params))
            score, _ = self._evaluate_params(param_dict)
            return -score  # Minimize negative score (maximize score)

        space = [Integer(min(v), max(v), name=k) for k, v in self.param_dict.items()]

        res = gp_minimize(
            objective,
            space,
            n_calls=50,
            random_state=0,
        )

        best_params = dict(zip(self.param_dict.keys(), res.x))
        best_score = -res.fun

        return TunedResult(
            best_params=best_params,
            best_score=best_score,
            results=[],  # We don't have detailed results for Bayesian optimization
        )

    def _flaml_optimization(self) -> TunedResult:

        def objective(config):
            print("config")
            print(config)
            score, result = self._evaluate_params(config)
            return {"score": score, "result": result}

        search_space = {
            k: (
                tune.choice(v["values"])
                if isinstance(v.get("values"), list)
                else tune.uniform(v["min"], v["max"])
            )
            for k, v in self.param_dict.items()
        }

        algo = BlendSearch(
            metric="score",
            mode="max",
            points_to_evaluate=(
                [self.current_param_dict] if self.current_param_dict else None
            ),
        )

        analysis = tune.run(
            objective,
            config=search_space,
            num_samples=self.num_simulations,
            search_alg=algo,
            verbose=self.show_progress,
        )

        best_trial = analysis.best_trial
        best_params = best_trial.config
        run_results = [
            RunResult(
                score=trial.last_result["score"],
                params=trial.config,
                metadata=trial.last_result["result"],
            )
            for trial in analysis.trials
        ]

        best_idx = run_results.index(
            next(result for result in run_results if result.params == best_params)
        )

        return TunedResult(run_results=run_results, best_idx=best_idx)

    def _evaluate_params(
        self, param_dict: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        # Use the enhanced function to evaluate the parameters
        result = self.enhanced_function(
            attack_params=param_dict,
            attack_llm=self.attack_llm,
            target_llm=self.target,
            evaluator_llm=self.evaluator,
            logger=self.fixed_param_dict["logger"],
            args=self.fixed_param_dict["args"],
            top_p=param_dict.get("top_p", 1.0),
            temperature=param_dict.get("temperature", 0.7),
            flavor="default",
        )
        print(result)
        avg_score = result["score"]
        return avg_score, result

    def save_results_table(self, results: pd.DataFrame, filepath: str):
        results.to_csv(filepath, index=False)

    def save_graphs(self, results: pd.DataFrame, output_dir: str):
        import matplotlib.pyplot as plt
        import os
        import json

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

        # Plotting: Cost vs. Hyperparameters
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

        # Clean up plots to avoid overlap in subsequent uses
        plt.clf()

        print("Graphs have been saved successfully.")
