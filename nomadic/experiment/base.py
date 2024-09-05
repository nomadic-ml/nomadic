from datetime import datetime
from enum import Enum
from pathlib import Path
import traceback
from typing import Any, Dict, List, Optional, Callable, Set, Union, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator

from llama_index.core.evaluation import BaseEvaluator
from llama_index.core.llms import CompletionResponse
from llama_index.core.base.response.schema import Response

from nomadic.client import get_client, NomadicClient
from nomadic.model import OpenAIModel, TogetherAIModel, SagemakerModel
from nomadic.result import RunResult, ExperimentResult
from nomadic.tuner.base import BaseParamTuner
from nomadic.util import is_ray_installed

from nomadic.experiment.prompt_tuning import PromptTuner, custom_evaluate

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

import time
import random
from functools import wraps


def retry_with_exponential_backoff(
    max_retries=5, base_delay=1, max_delay=300, exceptions=(Exception,)
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        raise e
                    delay = min(
                        max_delay, (base_delay * (2**attempt)) + random.uniform(0, 1)
                    )
                    print(
                        f"Exception occurred: {str(e)}. Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


class ExperimentStatus(str, Enum):
    not_started = "not_started"
    running = "running"
    finished_success = "finished_success"
    finished_error = "finished_error"


class ExperimentMode(str, Enum):
    train = "training"
    fine_tune = "fine_tuning"
    inference = "inference"


class Experiment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    params: Optional[Set[str]] = Field(
        default=set(), description="The set parameters to tune in experiment runs."
    )
    evaluation_dataset: Optional[List[Dict]] = Field(
        default=None,
        description="Evaluation dataset in dictionary format.",
    )
    param_fn: Optional[Callable[[Dict[str, Any]], Any]] = Field(
        default=None, description="Function to run with parameters."
    )
    model: Optional[Any] = Field(default=None, description="Model to run experiment")
    evaluator: Optional[Union[BaseEvaluator, Callable, Dict[str, Any]]] = Field(
        default=None,
        description="Evaluator of experiment (BaseEvaluator instance, callable, or dictionary)",
    )
    tuner: Optional[Any] = Field(default=None, description="Base Tuner")
    fixed_param_dict: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional dictionary of fixed hyperparameter values.",
    )
    current_param_dict: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional dictionary of current hyperparameter values.",
    )
    search_method: Optional[str] = Field(
        default="grid", description="Tuner search option. Can be: [grid, bayesian]"
    )
    start_datetime: Optional[datetime] = Field(
        default=None, description="Start datetime."
    )
    end_datetime: Optional[datetime] = Field(default=None, description="End datetime.")
    experiment_result: Optional[ExperimentResult] = Field(
        default=None, description="Experiment result of Experiment"
    )
    all_experiment_results: Optional[List[ExperimentResult]] = Field(
        default=[],
        description="All expeirment results of experiment (including from Workspace)",
    )
    experiment_status: Optional[ExperimentStatus] = Field(
        default=ExperimentStatus("not_started"),
        description="Current status of Experiment",
    )
    experiment_status_message: Optional[str] = Field(
        default="",
        description="Detailed description of Experiment status during error.",
    )
    enable_logging: bool = Field(
        default=True,
        description="Flag to enable or disable print logging.",
    )
    user_prompt_request: Optional[str] = Field(
        default="",
        description="User request for GPT prompt.",
    )
    num_samples: Optional[int] = Field(
        default=-1,
        description="Number of HP tuning samples to run. Only active for FLAML",
    )
    use_flaml_library: Optional[bool] = Field(
        default=False,
        description="Whether to use FLAML as parameter tuning library. If False, Ray Tune will be used.",
    )
    use_ray_backend: Optional[bool] = Field(
        default=False,
        description="Whether to use Ray Tune as parameter tuning backend. If False, Optuna will be used.",
    )
    results_filepath: Optional[str] = Field(
        default=None, description="Path of outputting tuner run results."
    )
    name: str = Field(default="my experiment", description="Name of experiment")
    client_id: Optional[str] = Field(
        default=None, description="ID of Experiment in Workspace"
    )
    # client: Optional[Client] = Field(
    #     default=None, description="Client to use for synching experiments."
    # )

    @field_validator("tuner")
    def check_tuner_class(cls, value):
        if value is not None and not isinstance(value, BaseParamTuner):
            raise ValueError("tuner must be a subclass of BaseParamTuner")
        return value

    def model_post_init(self, ctx):
        if self.search_method not in ("grid", "bayesian"):
            raise ValueError(
                f"Selected Experiment search_method `{self.search_method}` is not valid."
            )
        # TODO: Fix logic w.r.t. having either one experiment result per experiment, or having multiple experiment results
        if self.all_experiment_results and not self.experiment_result:
            self.experiment_result = self.all_experiment_results[0]
        if not self.client_id:  # Check this flag before registering
            nomadic_client: NomadicClient = get_client()
            if nomadic_client.auto_sync_enabled:
                nomadic_client.experiments.register(self)

    def run(self, param_dict: Dict[str, Any]) -> ExperimentResult:
        def _get_responses(
            type_safe_param_values: Tuple[Dict[str, Any], Dict[str, Any]]
        ) -> Tuple[List[str], List[str], List[str], List[str]]:
            all_pred_responses, all_eval_qs, all_ref_responses, all_prompt_variants = (
                [],
                [],
                [],
                [],
            )

            openai_params, prompt_tuning_params = type_safe_param_values

            # Initialize PromptTuner with the current parameters
            prompt_tuner = PromptTuner(
                prompting_approaches=[
                    prompt_tuning_params.get("prompt_tuning_approach", "None")
                ],
                prompt_complexities=[
                    prompt_tuning_params.get("prompt_tuning_complexity", "None")
                ],
                prompt_focuses=[
                    prompt_tuning_params.get("prompt_tuning_focus", "None")
                ],
                enable_logging=self.enable_logging,
            )

            # Generate prompt variants using PromptTuner
            prompt_variants = prompt_tuner.generate_prompt_variants(
                self.user_prompt_request,
                (
                    self.model.api_keys.get("OPENAI_API_KEY")
                    if hasattr(self.model, "api_keys")
                    else None
                ),
            )

            for i, prompt_variant in enumerate(prompt_variants):
                if self.enable_logging:
                    print(f"\nProcessing prompt variant {i+1}/{len(prompt_variants)}")
                    print(
                        f"Prompt variant: {prompt_variant[:200]}..."
                    )  # Print part of the variant

                pred_responses, eval_qs, ref_responses = [], [], []
                if not self.evaluation_dataset:
                    full_prompt = prompt_variant
                    completion_response: CompletionResponse = self.model.run(
                        prompt=full_prompt,
                        parameters=openai_params,
                    )
                    pred_response = self._extract_response(completion_response)
                    pred_responses.append(pred_response)
                    eval_qs.append(full_prompt)
                    ref_responses.append(None)
                    all_prompt_variants.append(prompt_variant)
                    if self.enable_logging:
                        print(f"Response: {pred_response[:100]}...")
                else:
                    for j, example in enumerate(self.evaluation_dataset):
                        if self.enable_logging:
                            print(
                                f"Processing example {j+1}/{len(self.evaluation_dataset)}"
                            )
                        full_prompt = self._construct_prompt(prompt_variant, example)
                        completion_response: CompletionResponse = self.model.run(
                            prompt=full_prompt,
                            parameters=openai_params,
                        )
                        pred_response = self._extract_response(completion_response)
                        pred_responses.append(pred_response)
                        eval_qs.append(full_prompt)
                        ref_responses.append(example.get("Answer", None))
                        all_prompt_variants.append(prompt_variant)
                        if self.enable_logging:
                            print(f"Response: {pred_response[:100]}...")
                all_pred_responses.extend(pred_responses)
                all_eval_qs.extend(eval_qs)
                all_ref_responses.extend(ref_responses)
            return (
                all_pred_responses,
                all_eval_qs,
                all_ref_responses,
                all_prompt_variants,
            )

        def _default_param_function(param_values: Dict[str, Any]) -> RunResult:
            if self.enable_logging:
                print("\nStarting new experiment run with parameters:")
                for param, value in param_values.items():
                    print(f"{param}: {value}")

            type_safe_param_values = self._enforce_param_types(param_values)
            pred_responses, eval_qs, ref_responses, prompt_variants = _get_responses(
                type_safe_param_values
            )
            eval_results = self._evaluate_responses(pred_responses, ref_responses)
            mean_score = self._calculate_mean_score(eval_results)

            if self.enable_logging:
                print(f"\nExperiment run completed. Mean score: {mean_score}")

            metadata = {
                "Answers": pred_responses,
                "Ground Truth": ref_responses,
                "Custom Evaluator Results": eval_results,
                "Full Prompts": eval_qs,
                "Prompt Variants": prompt_variants,
                "Prompt Parameters": {
                    k: v
                    for k, v in param_values.items()
                    if k
                    in [
                        "prompt_tuning_approach",
                        "prompt_tuning_complexity",
                        "prompt_tuning_focus",
                    ]
                },
            }

            if hasattr(self, "evaluation_metrics"):
                metadata["Evaluation Metrics"] = self.evaluation_metrics

            return RunResult(
                score=mean_score,
                params=param_values,
                metadata=metadata,
            )

        is_error = False
        self.experiment_status = ExperimentStatus("running")
        self.start_datetime = datetime.now()
        experiment_result: ExperimentResult = None
        try:
            if self.enable_logging:
                print("Setting up tuner...")
            self._setup_tuner(param_dict, _default_param_function)
            if self.enable_logging:
                print("Starting experiment...")
            experiment_result = self.tuner.fit()
        except Exception as e:
            is_error = True
            self.experiment_status_message = self._format_error_message(e)
            if self.enable_logging:
                print(f"Error occurred: {self.experiment_status_message}")

        self.end_datetime = datetime.now()
        self.experiment_status = self._determine_experiment_status(is_error)
        self.experiment_result = (
            experiment_result or self._create_default_experiment_result(param_dict)
        )
        if self.enable_logging:
            print(f"\nExperiment completed. Status: {self.experiment_status}")
        nomadic_client: NomadicClient = get_client()
        if nomadic_client.auto_sync_enabled:
            nomadic_client.experiments.register(self)
        return self.experiment_result

    def _construct_prompt(self, prompt_variant: str, example: Dict[str, str]) -> str:
        # Use the prompt variant as the base, which should already include any tuning modifications
        prompt = prompt_variant

        # Add example-specific information
        prompt += f"\n\nContext: {example['Context']}"
        prompt += f"\nInstruction: {example['Instruction']}"
        prompt += f"\nQuestion: {example['Question']}"

        return prompt

    def _extract_response(self, completion_response: CompletionResponse) -> str:
        if isinstance(self.model, OpenAIModel) or isinstance(
            self.model, TogetherAIModel
        ):
            return completion_response.text
        elif isinstance(self.model, SagemakerModel):
            return completion_response.raw["Body"]
        else:
            raise NotImplementedError("Unsupported model type")

    def _enforce_param_types(
        self, param_values: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        openai_params = {}
        prompt_tuning_params = {}
        for param, val in param_values.items():
            if param in self.model.hyperparameters.default:
                openai_params[param] = self.model.hyperparameters.default[param][
                    "type"
                ](val)
            else:
                prompt_tuning_params[param] = val
        return openai_params, prompt_tuning_params

    def _evaluate_responses(
        self, pred_responses: List[str], ref_responses: List[str]
    ) -> List[Any]:
        eval_results = []
        if self.evaluator:
            for pred, ref in zip(pred_responses, ref_responses):
                if isinstance(self.evaluator, dict):
                    if self.evaluator.get("method") == "custom_evaluate":
                        evaluation_metrics = self.evaluator.get(
                            "evaluation_metrics", []
                        )
                        if not evaluation_metrics:
                            raise ValueError(
                                "evaluation_metrics must be provided when using custom_evaluate"
                            )
                        for metric in evaluation_metrics:
                            if isinstance(metric, str):
                                metric = {"metric": metric, "weight": 1.0}
                            elif isinstance(metric, dict) and "weight" not in metric:
                                metric["weight"] = 1.0
                        openai_api_key = (
                            self.model.api_keys.get("OPENAI_API_KEY")
                            if hasattr(self.model, "api_keys")
                            else None
                        )
                        eval_results.append(
                            custom_evaluate(pred, evaluation_metrics, openai_api_key)
                        )
                    else:
                        raise ValueError("Invalid evaluator method")
                elif callable(self.evaluator):
                    eval_results.append(self.evaluator(pred, ref))
                elif isinstance(self.evaluator, BaseEvaluator):
                    eval_results.append(
                        {
                            "score": self.evaluator.evaluate_response(
                                response=Response(pred), reference=ref
                            ).score
                        }
                    )
                else:
                    raise ValueError("Invalid evaluator type")
        return eval_results

    def _calculate_mean_score(self, eval_results: List[Any]) -> float:
        scores = []
        for result in eval_results:
            if "scores" in result and "Overall score" in result["scores"]:
                scores.append(result["scores"]["Overall score"])
            elif "overall_score" in result and result["overall_score"] != 0:
                scores.append(result["overall_score"])
            elif "score" in result:
                scores.append(result["score"])
            elif hasattr(result, "score"):
                scores.append(result.score)

        return sum(scores) / len(scores) if scores else 0

    def _setup_tuner(self, param_dict: Dict[str, Any], param_function: Callable):
        if not self.tuner:
            if self.enable_logging:
                print("\nSetting up tuner...")
            if self.use_flaml_library:
                from nomadic.tuner import FlamlParamTuner

                self.tuner = FlamlParamTuner(
                    param_fn=param_function,
                    param_dict=param_dict,
                    search_method=self.search_method,
                    fixed_param_dict=self.fixed_param_dict,
                    current_param_dict=self.current_param_dict,
                    show_progress=self.enable_logging,
                    use_ray=self.use_ray_backend,
                    num_samples=self.num_samples,
                )
            elif is_ray_installed():
                from nomadic.tuner.ray import RayTuneParamTuner
                from ray import tune as ray_tune

                # Convert param_dict to Ray Tune format
                ray_param_dict = {}
                for param, value in param_dict.items():
                    if isinstance(value, list):
                        ray_param_dict[param] = ray_tune.choice(value)
                    else:
                        ray_param_dict[param] = value

                self.tuner = RayTuneParamTuner(
                    param_fn=param_function,
                    param_dict=ray_param_dict,  # Use the converted ray_param_dict
                    search_method=self.search_method,
                    fixed_param_dict=self.fixed_param_dict,
                    current_param_dict=self.current_param_dict,
                    show_progress=self.enable_logging,
                )
            else:
                raise NotImplemented(
                    "Only FLAML and Ray Tune are supported as tuning backends."
                )

        if param_dict:
            self.tuner.param_dict = param_dict
        if self.param_fn:
            self.tuner.param_fn = self.param_fn
        if self.fixed_param_dict:
            self.tuner.fixed_param_dict = self.fixed_param_dict
        if self.results_filepath:
            self.tuner.results_filepath = self.results_filepath
        self.tuner.show_progress = self.enable_logging

    def _format_error_message(self, exception: Exception) -> str:
        return f"Exception: {str(exception)}\n\nStack Trace:\n{traceback.format_exc()}"

    def _determine_experiment_status(self, is_error: bool) -> ExperimentStatus:
        return (
            ExperimentStatus("finished_success")
            if not is_error
            else ExperimentStatus("finished_error")
        )

    def _create_default_experiment_result(
        self, param_dict: Dict[str, Any]
    ) -> ExperimentResult:
        return ExperimentResult(
            hp_search_space=param_dict,
            run_results=[RunResult(score=-1, params={}, metadata={})],
        )

    def save_experiment(self, folder_path: Path):
        file_name = (
            f"/experiment_{self.start_datetime.strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )
        with open(folder_path + file_name, "w") as file:
            file.write(self.model_dump_json(exclude=("model", "evaluator")))

    def visualize_results(self, experiment_result, graphs_to_include=None):
        if not experiment_result or not experiment_result.run_results:
            if self.enable_logging:
                print("No results to visualize.")
            return

        if graphs_to_include is None:
            graphs_to_include = [
                "overall_score_distribution",
                "metric_score_distributions",
                "parameter_relationships",
                "summary_statistics",
                "correlation_heatmap",
                "parameter_combination_heatmap",
            ]

        # Extract data from experiment results
        data = self._extract_data_from_results(experiment_result)

        # Create and clean DataFrame
        df = self._create_and_clean_dataframe(data)

        # Separate numeric and categorical columns
        numeric_cols, categorical_cols = self._separate_column_types(df)

        if "overall_score_distribution" in graphs_to_include:
            self._plot_score_distribution(
                df, "overall_score", "Distribution of Overall Scores"
            )

        if "metric_score_distributions" in graphs_to_include:
            for metric in data["all_metric_scores"].keys():
                if metric in df.columns:
                    self._plot_score_distribution(
                        df, metric, f"Distribution of {metric} Scores"
                    )

        if "parameter_relationships" in graphs_to_include:
            self._visualize_parameter_relationships(df, numeric_cols, categorical_cols)

        if "summary_statistics" in graphs_to_include:
            self._print_summary_statistics(df, data["all_metric_scores"])

        if "correlation_heatmap" in graphs_to_include:
            self._create_correlation_heatmap(df, numeric_cols)

        if "parameter_combination_heatmap" in graphs_to_include:
            self._create_parameter_combination_heatmap(experiment_result)

    def _extract_data_from_results(self, experiment_result):
        all_scores = []
        all_params = []
        all_metric_scores = {}

        for run in experiment_result.run_results:
            metadata = run.metadata
            all_scores.append(run.score)
            all_params.append(metadata.get("Prompt Parameters", {}))

            eval_result = metadata.get("Custom Evaluator Results", [{}])[0]
            if "scores" in eval_result:
                for metric, score in eval_result["scores"].items():
                    all_metric_scores.setdefault(metric, []).append(score)

        return {
            "all_scores": all_scores,
            "all_params": all_params,
            "all_metric_scores": all_metric_scores,
        }

    def _create_and_clean_dataframe(self, data):
        df = pd.DataFrame(data["all_params"])
        df["overall_score"] = data["all_scores"]

        for metric, scores in data["all_metric_scores"].items():
            df[metric] = scores

        return df.dropna(axis=1, how="any").dropna()

    def _separate_column_types(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        return numeric_cols, categorical_cols

    def _plot_score_distribution(self, df, column, title):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=column, kde=True)
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

    def _visualize_parameter_relationships(self, df, numeric_cols, categorical_cols):
        for col in categorical_cols:
            if col != "overall_score":
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=col, y="overall_score", data=df)
                plt.title(f"Overall Score Distribution by {col}")
                plt.ylabel("Overall Score")
                plt.xticks(rotation=45)
                plt.show()

        for col in numeric_cols:
            if col != "overall_score":
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=col, y="overall_score", data=df)
                plt.title(f"Overall Score vs {col}")
                plt.ylabel("Overall Score")
                plt.show()

    def _print_summary_statistics(self, df, all_metric_scores):
        print("Score Summary Statistics:")
        print(df[["overall_score"] + list(all_metric_scores.keys())].describe())

        print("\nTop 5 Performing Parameter Combinations:")
        top_5 = df.sort_values("overall_score", ascending=False).head()
        print(top_5.to_string(index=False))

    def _create_correlation_heatmap(self, df, numeric_cols):
        plt.figure(figsize=(12, 10))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap of Numeric Parameters and Scores")
        plt.show()

    def _create_parameter_combination_heatmap(self, experiment_result):
        run_results = experiment_result.run_results

        data = []
        param_names = set()
        for run in run_results:
            row = run.params.copy()
            param_names.update(row.keys())
            row["overall_score"] = run.score

            if (
                "Custom Evaluator Results" in run.metadata
                and run.metadata["Custom Evaluator Results"]
            ):
                scores = run.metadata["Custom Evaluator Results"][0].get("scores", {})
                if isinstance(scores, dict):
                    row.update(scores)
                elif isinstance(scores, (int, float)):
                    row["single_score"] = scores

            data.append(row)

        df = pd.DataFrame(data)

        # Remove columns with less than 50% of entries
        threshold = len(df) * 0.5
        df = df.dropna(axis=1, thresh=threshold)

        # Separate hyperparameters and score columns
        hyperparameter_cols = list(param_names)
        score_cols = [
            col
            for col in df.columns
            if col not in hyperparameter_cols and col != "overall_score"
        ]
        score_cols = ["overall_score"] + score_cols  # Ensure overall_score is first

        # Create param_combination string
        def format_param_combination(row):
            parts = []
            for k in hyperparameter_cols:
                parts.append(f"{k}: {row[k]}")
            return " | ".join(parts)

        df["param_combination"] = df.apply(format_param_combination, axis=1)

        # Group by parameter combination
        grouped_df = df.groupby("param_combination")[score_cols].mean()
        grouped_df["num_simulations"] = df.groupby("param_combination").size()

        # Sort by the overall score column
        grouped_df = grouped_df.sort_values("overall_score", ascending=False)

        # Calculate overall averages
        metric_averages = grouped_df[score_cols].mean()

        # Print average scores for each metric
        print("Average Scores Across All Parameter Sets:")
        for metric, score in metric_averages.items():
            print(f"{metric}: {score:.2f}")

        # Add overall average scores as a separate row
        avg_row = pd.DataFrame([metric_averages], index=["AVERAGE"])
        grouped_df = pd.concat([grouped_df, avg_row])

        # Transpose the DataFrame to swap rows and columns
        grouped_df_t = grouped_df.transpose()
        # Create heatmap
        fig, ax = plt.subplots(
            figsize=(len(grouped_df) * 1.2, 10)
        )  # Adjust figure size

        sns.heatmap(
            grouped_df_t,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            cbar_kws={"label": "Score"},
            ax=ax,
            linewidths=0.5,
            square=True,
        )
        plt.title("Heatmap of Scores for Each Parameter Combination", pad=20)
        plt.ylabel("Metric", labelpad=20)
        plt.xlabel("Parameter Combination", labelpad=20)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        # Adjust layout
        plt.tight_layout()

        # Increase font size for better readability
        plt.setp(ax.get_xticklabels(), fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=10)

        # Add gridlines
        ax.set_xticks(np.arange(grouped_df.shape[0]) + 0.5, minor=False)
        ax.xaxis.grid(True, which="major", linestyle="-", linewidth=0.5, color="white")

        plt.show()

        return grouped_df

    def test_significance(self, experiment_result: ExperimentResult, n: int = 5):
        if not experiment_result or not experiment_result.run_results:
            print("No results to test for significance.")
            return

        scores = [run.score for run in experiment_result.run_results]
        sorted_scores = sorted(scores, reverse=True)

        if len(sorted_scores) <= n:
            print(
                f"Not enough data points to perform significance test. Need more than {n} results."
            )
            return

        top_n_scores = sorted_scores[:n]
        rest_scores = sorted_scores[n:]

        statistic, p_value = stats.mannwhitneyu(
            top_n_scores, rest_scores, alternative="two-sided"
        )

        print(f"Mann-Whitney U test results:")
        print(f"Comparing top {n} scores against the rest")
        print(f"U-statistic: {statistic}")
        print(f"p-value: {p_value}")

        alpha = 0.05
        if p_value < alpha:
            print(
                f"The difference between the top {n} parameter combinations and the rest is statistically significant (p < {alpha})."
            )
        else:
            print(
                f"There is no statistically significant difference between the top {n} parameter combinations and the rest (p >= {alpha})."
            )

        mean_top_n, mean_rest = np.mean(top_n_scores), np.mean(rest_scores)
        pooled_std = np.sqrt(
            (np.std(top_n_scores, ddof=1) ** 2 + np.std(rest_scores, ddof=1) ** 2) / 2
        )
        cohen_d = (mean_top_n - mean_rest) / pooled_std

        print(f"\nEffect size (Cohen's d): {cohen_d:.2f}")
        if abs(cohen_d) < 0.2:
            print("The effect size is small.")
        elif abs(cohen_d) < 0.5:
            print("The effect size is medium.")
        else:
            print("The effect size is large.")

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=[top_n_scores, rest_scores], orient="h")
        plt.title(f"Comparison of Top {n} Scores vs Rest")
        plt.xlabel("Score")
        plt.yticks([0, 1], [f"Top {n}", "Rest"])
        plt.show()

        return {
            "statistic": statistic,
            "p_value": p_value,
            "cohen_d": cohen_d,
            "top_n_scores": top_n_scores,
            "rest_scores": rest_scores,
        }

    def wrap_text(self, text, width):
        # Wrap text to fit within the given width
        from textwrap import wrap

        return wrap(text, width)

    def create_run_results_table(
        self,
        experiment_result,
        max_rows=None,
        max_prompt_length=100,
        max_answer_length=500,
    ):
        data = []
        for run_result in experiment_result.run_results:
            row = {
                "Score": f"{run_result.score:.2f}",
                "Full Prompt": self._wrap_text(
                    run_result.metadata.get("Prompt Variants", [""])[0],
                    max_prompt_length,
                ),
                "Answer": self._wrap_text(
                    run_result.metadata.get("Answers", [""])[0], max_answer_length
                ),
            }
            row.update(run_result.metadata.get("Prompt Parameters", {}))
            data.append(row)

        df = pd.DataFrame(data)

        if max_rows is not None:
            df = df.head(max_rows)

        return df

    def _wrap_text(self, text, max_length):
        if len(text) <= max_length:
            return text

        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > max_length:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + 1

        if current_line:
            lines.append(" ".join(current_line))

        return "\n".join(lines)
