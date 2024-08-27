from datetime import datetime
from enum import Enum
from pathlib import Path
import traceback
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator

from llama_index.core.evaluation import BaseEvaluator
from llama_index.core.llms import CompletionResponse
from llama_index.core.base.response.schema import Response

# from nomadic.client import Client
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


class ExperimentStatus(Enum):
    not_started = "not_started"
    running = "running"
    finished_success = "finished_success"
    finished_error = "finished_error"


class ExperimentMode(Enum):
    train = "training"
    fine_tune = "fine_tuning"
    inference = "inference"


class Experiment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    param_dict: Optional[Dict[str, Any]] = Field(
        default={}, description="A dictionary of parameters to iterate over."
    )
    evaluation_dataset: Optional[List[Dict]] = Field(
        default=list({}),
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
    prompts: Optional[List[str]] = Field(
        default=PromptTuner(enable_logging=False),
        description="Optional list of prompts to use in the experiment.",
    )
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
        default=None, description="Tuned result of Experiment"
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
    prompt_tuner: Optional[PromptTuner] = Field(
        default=None,
        description="PromptTuner instance for generating prompt variants.",
    )
    user_prompt_request: Optional[str] = Field(
        default="",
        description="User request for GPT prompt.",
    )
    num_example_prompts: Optional[int] = Field(
        default=0,
        description="Number of example prompts to include for few-shot prompting.",
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
    name: Optional[str] = Field(default=None, description="Name of experiment")
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

    def run(self) -> ExperimentResult:
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
                    prompt_tuning_params.get("prompt_tuning_approach", "zero-shot")
                ],
                prompt_complexities=[
                    prompt_tuning_params.get("prompt_tuning_complexity", "simple")
                ],
                prompt_focuses=[
                    prompt_tuning_params.get("prompt_tuning_focus", "fact extraction")
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
        result = None
        try:
            # if self.enable_logging:
            #     print("Setting up client...")
            #     self._setup_client()
            if self.enable_logging:
                print("Setting up tuner...")
            self._setup_tuner(_default_param_function)
            if self.enable_logging:
                print("Starting experiment...")
            result = self.tuner.fit()
        except Exception as e:
            is_error = True
            self.experiment_status_message = self._format_error_message(e)
            if self.enable_logging:
                print(f"Error occurred: {self.experiment_status_message}")

        self.end_datetime = datetime.now()
        self.experiment_status = self._determine_experiment_status(is_error)
        self.experiment_result = result or self._create_default_experiment_result()
        if self.enable_logging:
            print(f"\nExperiment completed. Status: {self.experiment_status}")
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
                        self.evaluator.evaluate_response(
                            response=Response(pred), reference=ref
                        )
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
            elif hasattr(result, "score"):
                scores.append(result.score)

        return sum(scores) / len(scores) if scores else 0

    def _setup_tuner(self, param_function: Callable):
        if not self.tuner:
            if self.enable_logging:
                print("\nSetting up tuner...")
            if self.use_flaml_library:
                from nomadic.tuner import FlamlParamTuner

                self.tuner = FlamlParamTuner(
                    param_fn=param_function,
                    param_dict=self.param_dict,
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
                for param, value in self.param_dict.items():
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

        if self.param_fn:
            self.tuner.param_fn = self.param_fn
        if self.param_dict:
            self.tuner.param_dict = self.param_dict
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

    def _create_default_experiment_result(self) -> ExperimentResult:
        return ExperimentResult(
            run_results=[RunResult(score=-1, params={}, metadata={})]
        )

    def save_experiment(self, folder_path: Path):
        file_name = (
            f"/experiment_{self.start_datetime.strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )
        with open(folder_path + file_name, "w") as file:
            file.write(self.model_dump_json(exclude=("model", "evaluator")))

    def visualize_results(self, tuned_result, graphs_to_include=None):
        if not tuned_result or not tuned_result.run_results:
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
        data = self._extract_data_from_results(tuned_result)

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
            self._create_parameter_combination_heatmap(df, data["all_metric_scores"])

    def _extract_data_from_results(self, tuned_result):
        all_scores = []
        all_params = []
        all_metric_scores = {}

        for run in tuned_result.run_results:
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

    def _create_parameter_combination_heatmap(self, tuned_result):
        run_results = tuned_result.run_results

        # Function to automatically abbreviate parameter names and values
        def auto_abbreviate(s):
            if isinstance(s, str):
                words = s.split("_")
                return "".join(word[0] for word in words)
            return str(s)

        # Extract data from run_results
        data = []
        for run in run_results:
            row = run.params.copy()  # Start with all params
            row["overall_score"] = run.score

            # Extract custom evaluator scores if available
            if (
                "Custom Evaluator Results" in run.metadata
                and run.metadata["Custom Evaluator Results"]
            ):
                scores = run.metadata["Custom Evaluator Results"][0].get("scores", {})
                row.update(scores)

            data.append(row)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Remove columns with less than 50% of entries
        threshold = len(df) * 0.5
        df = df.dropna(axis=1, thresh=threshold)

        # Create a string representation of the parameter combination for grouping
        df["param_combination"] = df.apply(
            lambda row: tuple(row.drop("overall_score")), axis=1
        )

        # Group by parameter combination and calculate mean for each metric
        grouped_df = df.groupby("param_combination").agg(
            {col: "mean" for col in df.columns if col != "param_combination"}
        )
        grouped_df["num_simulations"] = df.groupby("param_combination").size()

        # Create abbreviated parameter combination column
        grouped_df["param_combination_abbr"] = grouped_df.index.map(
            lambda x: " | ".join(
                f"{auto_abbreviate(k)}:{auto_abbreviate(v)}" for k, v in dict(x).items()
            )
        )

        # Select metric columns
        metric_columns = [
            col
            for col in grouped_df.columns
            if col not in ["num_simulations", "param_combination_abbr"]
        ]
        heatmap_data = grouped_df[metric_columns + ["num_simulations"]]

        # Function to combine similar columns
        def combine_similar_columns(df, threshold=0.95):
            # Compute the correlation matrix
            corr = df.drop("num_simulations", axis=1).corr().abs()
            corr = corr.fillna(0)
            linkage = hierarchy.linkage(pdist(corr), method="complete")
            clusters = hierarchy.fcluster(
                linkage, t=1 - threshold, criterion="distance"
            )

            new_df = pd.DataFrame()
            for cluster_id in np.unique(clusters):
                cols = df.drop("num_simulations", axis=1).columns[
                    clusters == cluster_id
                ]
                if len(cols) == 1:
                    new_df[cols[0]] = df[cols[0]]
                else:
                    new_col_name = " / ".join(cols)
                    new_df[new_col_name] = df[cols].mean(axis=1)

            new_df["num_simulations"] = df["num_simulations"]
            return new_df

        # Combine similar columns
        heatmap_data = combine_similar_columns(heatmap_data)

        # Ensure there's an overall score column
        if "overall_score" not in heatmap_data.columns:
            heatmap_data["overall_score"] = heatmap_data.mean(axis=1)

        # Sort by the overall score column
        heatmap_data = heatmap_data.sort_values("overall_score", ascending=False)

        # Calculate overall averages
        metric_averages = heatmap_data.mean()

        # Print average scores for each metric
        print("Average Scores Across All Parameter Sets:")
        for metric, score in metric_averages.items():
            print(f"{metric}: {score:.2f}")

        # Add overall average scores as a separate row
        avg_row = pd.DataFrame([metric_averages], index=["AVERAGE"])
        heatmap_data = pd.concat([heatmap_data, avg_row])

        # Create heatmap
        plt.figure(figsize=(15, len(heatmap_data) * 0.4 + 1))
        sns.heatmap(
            heatmap_data.drop("num_simulations", axis=1),
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            cbar_kws={"label": "Score"},
        )
        plt.title("Heatmap of Scores for Each Parameter Combination")
        plt.ylabel("Parameter Combination")
        plt.xlabel("Metric")
        plt.xticks(rotation=45, ha="right")

        # Use abbreviated parameter combinations for y-axis labels
        y_labels = grouped_df["param_combination_abbr"].tolist() + ["AVERAGE"]
        plt.yticks(range(len(y_labels)), y_labels, rotation=0, ha="right")

        plt.tight_layout()
        plt.show()

        # Print num_simulations for each parameter combination
        print("\nNumber of Simulations for Each Parameter Combination:")
        for index, row in heatmap_data.iterrows():
            if index != "AVERAGE":
                print(
                    f"{grouped_df.loc[index, 'param_combination_abbr']}: {row['num_simulations']:.0f}"
                )
            else:
                print(f"AVERAGE: {row['num_simulations']:.2f}")

        return grouped_df

    def test_significance(self, tuned_result: ExperimentResult, n: int = 5):
        if not tuned_result or not tuned_result.run_results:
            print("No results to test for significance.")
            return

        scores = [run.score for run in tuned_result.run_results]
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

    def create_run_results_table(self, tuned_result, max_rows=None):
        # Define column widths
        score_width = 10
        prompt_width = 40
        answer_width = 40
        param_width = 15

        # Get all unique parameter names
        all_params = set()
        for run_result in tuned_result.run_results:
            all_params.update(run_result.metadata.get("Prompt Parameters", {}).keys())

        # Create header
        header = f"| {'Score':^{score_width}} | {'Full Prompt':<{prompt_width}} | {'Answer':<{answer_width}}"
        for param in all_params:
            header += f" | {param:<{param_width}}"
        header += " |"

        separator = f"+{'-' * (score_width + 2)}+{'-' * (prompt_width + 2)}+{'-' * (answer_width + 2)}"
        separator += f"+{'-' * (param_width + 2)}" * len(all_params) + "+"

        # Create table string
        table = [separator, header, separator]

        row_count = 0
        for run_result in tuned_result.run_results:
            if max_rows is not None and row_count >= max_rows:
                break

            score = f"{run_result.score:.2f}"
            prompt_variants = run_result.metadata.get("Prompt Variants", [])
            full_prompt = (
                self.wrap_text(prompt_variants[0][:1000], prompt_width)
                if prompt_variants
                else [""]
            )
            answer = self.wrap_text(
                run_result.metadata.get("Answers", [""])[0][:1000], answer_width
            )

            prompt_params = run_result.metadata.get("Prompt Parameters", {})

            max_lines = max(len(full_prompt), len(answer))
            for i in range(max_lines):
                if max_rows is not None and row_count >= max_rows:
                    break

                prompt_line = full_prompt[i] if i < len(full_prompt) else ""
                answer_line = answer[i] if i < len(answer) else ""
                row = f"| {score if i == 0 else ' ':^{score_width}} | {prompt_line:<{prompt_width}} | {answer_line:<{answer_width}}"
                for param in all_params:
                    param_value = prompt_params.get(param, "N/A")
                    row += f" | {param_value if i == 0 else ' ':<{param_width}}"
                row += " |"
                table.append(row)
                row_count += 1

            if row_count < max_rows or max_rows is None:
                table.append(separator)

        # Add ellipsis if there are more rows
        if (
            max_rows is not None
            and row_count >= max_rows
            and row_count < len(tuned_result.run_results)
        ):
            table.append(
                f"| {'...':^{score_width}} | {'...':^{prompt_width}} | {'...':^{answer_width}}"
                + f" | {'...':^{param_width}}" * len(all_params)
                + " |"
            )
            table.append(separator)

        return "\n".join(table)
