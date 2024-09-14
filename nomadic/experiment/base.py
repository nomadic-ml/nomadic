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

from nomadic.experiment.prompt_tuning import (
    PromptTuner,
    custom_evaluate,
    custom_evaluate_hallucination,
)

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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from itertools import combinations

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
    user_prompt_request: Optional[Union[str, List[str]]] = Field(
        default=[],
        description="User request for GPT prompt. Can be a string or a list of strings.",
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
    # client: Optional[Client] = Field(
    #     default=None, description="Client to use for synching experiments."
    # )

    client_id: Optional[str] = Field(
        default=None, description="ID of Experiment in Workspace"
    )
    num_simulations: int = Field(
        default=1,
        description="Number of simulations to run for each configuration.",
    )
    @field_validator('user_prompt_request')
    def validate_user_prompt_request(cls, v):
        if isinstance(v, str):
            return [v]
        elif isinstance(v, list):
            return v
        else:
            raise ValueError("user_prompt_request must be a string or a list of strings")


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
        if self.all_experiment_results and not self.experiment_result:
            self.experiment_result = self.all_experiment_results[0]
        if not self.client_id:
            nomadic_client: NomadicClient = get_client()
            if nomadic_client.auto_sync_enabled:
                nomadic_client.experiments.register(self)

    def _get_responses(
        self,  # Add 'self' here to make it an instance method
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
            prompt_tuning_approaches=[
                prompt_tuning_params.get("prompt_tuning_approach", "None")
            ],
            prompt_tuning_topics=[
                prompt_tuning_params.get("prompt_tuning_topic", "hallucination-detection")
            ],
            prompt_tuning_complexities=[
                prompt_tuning_params.get("prompt_tuning_complexity", "None")
            ],
            prompt_tuning_focuses=[
                prompt_tuning_params.get("prompt_tuning_focus", "None")
            ],
            enable_logging=self.enable_logging,
        )
        self.user_prompt_request = self.user_prompt_request * self.num_simulations

        # Generate prompt variants using PromptTuner
        prompt_variants = prompt_tuner.generate_prompt_variants(
            client=self.model,  # Pass the client or the required object
            user_prompt_request=self.user_prompt_request
        )
        for i, prompt_variant in enumerate(prompt_variants):
            if self.enable_logging:
                print(f"\nProcessing prompt variant {i+1}/{len(prompt_variants)}")
                print(f"Prompt variant: {prompt_variant[:200]}...")  # Print part of the variant

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
                        print(f"Processing example {j+1}/{len(self.evaluation_dataset)}")
                    full_prompt = self._construct_prompt(prompt_variant, example)
                    completion_response: CompletionResponse = self.model.run(
                        prompt=full_prompt,
                        parameters=openai_params,
                    )
                    pred_response = self._extract_response(completion_response)
                    pred_responses.append(pred_response)
                    eval_qs.append(full_prompt)
                    ref_responses.append(example.get("answer") or example.get("Answer", None))
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


    def run(self, param_dict: Dict[str, Any]) -> ExperimentResult:
        self.params = set(param_dict.keys())  # Set the params attribute

        def _default_param_function(param_values: Dict[str, Any]) -> RunResult:
            if self.enable_logging:
                print("\nStarting new experiment run with parameters:")
                for param, value in param_values.items():
                    print(f"{param}: {value}")

            all_scores = []
            all_metadata = []

            type_safe_param_values = self._enforce_param_types(param_values)

            (
                all_pred_responses,
                all_full_prompts,
                all_ref_responses,
                prompt_variants,
            ) = self._get_responses(type_safe_param_values)
            print(all_pred_responses,
                all_full_prompts,
                all_ref_responses,
                prompt_variants)
            if self.evaluation_dataset:
                eval_results = self._evaluate_responses(
                    all_pred_responses, all_ref_responses, self.evaluation_dataset
                )
            else:
                eval_results = self._evaluate_responses(
                    all_pred_responses, all_ref_responses
                )

            for result in eval_results:
                result["params"] = param_values

            mean_scores = self._calculate_mean_score(eval_results)

            current_param_key = str(
                tuple(sorted((k, param_values[k]) for k in self.params))
            )

            current_score = mean_scores.get(current_param_key, 0.0)

            all_scores.append(current_score)
            queries = [item.get('query', '') for item in self.evaluation_dataset] if self.evaluation_dataset else []

            metadata = {
                "Answers": all_pred_responses,
                "Ground Truth": all_ref_responses,
                "Custom Evaluator Results": eval_results,
                "Full Prompts": all_full_prompts,
                "Queries": queries,
                "Prompt Variants": prompt_variants,
                "Prompt Parameters": {
                    k: v
                    for k, v in param_values.items()
                    if k
                    in [
                        "prompt_tuning_approach",
                        "prompt_tuning_topic",
                        "prompt_tuning_complexity",
                        "prompt_tuning_focus",
                    ]
                },
                "All Mean Scores": {str(k): v for k, v in mean_scores.items()},
            }

            if hasattr(self, "evaluation_metrics"):
                metadata["Evaluation Metrics"] = self.evaluation_metrics

            all_metadata.append(metadata)

            final_score = sum(all_scores) / len(all_scores)

            if self.enable_logging:
                print(
                    f"\nAll simulations completed. Final average score: {final_score}"
                )

            import numpy as np

            return RunResult(
                score=final_score,
                params=param_values,
                metadata={
                    "Individual Simulation Results": all_metadata,
                    "All Simulation Scores": all_scores,
                    "Number of Simulations": self.num_simulations,
                    "Median Score": np.median(all_scores),
                    "Best Score": max(all_scores),
                    "Score Standard Deviation": np.std(all_scores),
                },
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

    def _construct_prompt(
        self,
        prompt_variant: str,
        example: Dict[str, str] = None,
        context: str = "",
        instruction: str = "",
        query: str = "",
        question: str = "",
        response: str = "",
        answer: str = ""
    ) -> str:
        # Determine values for query/question (interchangeable) and response/answer (irreplaceable)
        query_value = example.get("query", query) or query or question
        response_value = example.get("response", response) or response or answer

        # Prepare replacements dictionary
        replacements = {
            "[CONTEXT]": example.get("context", context) or context,
            "[QUERY]": query_value,
            "[INSTRUCTION]": example.get("instruction", instruction) or instruction,
            "[RESPONSE]": response_value,
        }

        # Replace placeholders in the prompt variant
        for placeholder, value in replacements.items():
            prompt_variant = prompt_variant.replace(placeholder, value)

        # Find and remove unused placeholders
        unused_placeholders = [
            key for key in replacements.keys() if key in prompt_variant
        ]
        for placeholder in unused_placeholders:
            prompt_variant = prompt_variant.replace(placeholder, "")

        return prompt_variant.strip()

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
        self,
        pred_responses: List[str],
        ref_responses: List[str],
        evaluation_dataset: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Any]:
        eval_results = []

        for pred, ref in zip(pred_responses, ref_responses):
            if self.evaluator:
                if isinstance(self.evaluator, dict):
                    method = self.evaluator.get("method")
                    if method == "custom_evaluate_hallucination":
                        if evaluation_dataset:
                            eval_result = custom_evaluate_hallucination(
                                pred, ref, evaluation_dataset
                            )
                        else:
                            eval_result = custom_evaluate_hallucination(pred, ref)
                        eval_result.update({"generated_answer": pred, "ground_truth": ref})
                        eval_results.append(eval_result)

                    elif method == "custom_evaluate":
                        evaluation_metrics = self.evaluator.get("evaluation_metrics", [])
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
                        eval_result = custom_evaluate(pred, evaluation_metrics, openai_api_key)
                        eval_results.append(eval_result)

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

    def _calculate_mean_score(
        self, eval_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        if not eval_results:
            print("Warning: No evaluation results provided.")
            return {}

        scores_by_params = {}

        # Check if custom evaluation logic is required
        use_custom_evaluate = (
            isinstance(self.evaluator, dict)
            and self.evaluator.get("method") == "custom_evaluate"
        )

        for result in eval_results:
            if use_custom_evaluate:
                # Handle custom evaluation logic
                if "scores" in result and isinstance(result["scores"], dict):
                    overall_score = result.get("overall_score")
                    if overall_score is not None and isinstance(overall_score, (int, float)):
                        params = result.get("params", {})
                        param_key = tuple(
                            sorted((k, v) for k, v in params.items() if k in self.params)
                        )

                        if param_key not in scores_by_params:
                            scores_by_params[param_key] = []
                        scores_by_params[param_key].append(overall_score)
                    else:
                        print(f"Warning: 'scores' present but 'overall_score' is missing or invalid: {result}")
                else:
                    print(f"Warning: Unexpected result format for custom evaluation: {result}")
                    continue
            else:
                # Standard evaluation logic
                if not isinstance(result, dict) or "score" not in result:
                    print(f"Warning: Unexpected result format: {result}")
                    continue

                score = result["score"]
                if not isinstance(score, (int, float)):
                    print(
                        f"Warning: Invalid score type: {type(score)}. Expected int or float."
                    )
                    continue

                params = result.get("params", {})
                param_key = tuple(
                    sorted((k, v) for k, v in params.items() if k in self.params)
                )

                if param_key not in scores_by_params:
                    scores_by_params[param_key] = []
                scores_by_params[param_key].append(score)

        # Calculate mean scores for all parameter combinations
        mean_scores = {
            str(param_key): sum(scores) / len(scores)
            for param_key, scores in scores_by_params.items()
        }

        print("Debug: Calculated mean_scores =", mean_scores)

        return mean_scores

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

                ray_param_dict = {}
                for param, value in param_dict.items():
                    if isinstance(value, list):
                        ray_param_dict[param] = ray_tune.choice(value)
                    else:
                        ray_param_dict[param] = value

                self.tuner = RayTuneParamTuner(
                    param_fn=param_function,
                    param_dict=ray_param_dict,
                    search_method=self.search_method,
                    fixed_param_dict=self.fixed_param_dict,
                    current_param_dict=self.current_param_dict,
                    show_progress=self.enable_logging,
                )
            else:
                raise NotImplementedError(
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

    def create_parameter_combination_heatmap(
        self,
        experiment_result,
        max_rows=None,
        max_prompt_length=100,
        max_answer_length=500,
    ):
        try:
            import ipywidgets as widgets
            from IPython.display import display, HTML
        except ModuleNotFoundError:
            print("Please run `pip install ...` and run in a .ipynb to view visualizations & heatmaps")
            return

        data = []
        for run_result in experiment_result.run_results:
            individual_results = run_result.metadata.get('Individual Simulation Results', [])
            for result in individual_results:
                # Extract data from the result
                answers = result.get('Answers', [])
                ground_truths = result.get('Ground Truth', [])
                full_prompts = result.get('Full Prompts', [])
                prompt_variants = result.get('Prompt Variants', [])
                queries = result.get('Queries', [])

                if not all([answers, ground_truths, full_prompts, prompt_variants, queries]):
                    if self.enable_logging:
                        print(f"Missing or empty data in result: {result}")
                    continue

                # Ensure all lists have the same length
                min_length = min(len(answers), len(ground_truths), len(full_prompts), len(queries))

                # Create a row for each prompt variant and corresponding data
                for i in range(min_length):
                    row = {
                        "Run Score": float(run_result.score),
                        "Prompt Variant": self._wrap_text(prompt_variants[i % len(prompt_variants)], max_prompt_length),
                        "Full Prompt": self._wrap_text(full_prompts[i], max_prompt_length),
                        "Generated Answer": self._wrap_text(answers[i], max_answer_length),
                        "Ground Truth": self._wrap_text(ground_truths[i], max_answer_length),
                        "Query": self._wrap_text(queries[i], max_prompt_length),
                    }
                    # Process parameters to ensure scalar values
                    processed_params = {}
                    for key, value in run_result.params.items():
                        # Handle NumPy arrays and scalars
                        if isinstance(value, (list, np.ndarray)):
                            if len(value) == 1:
                                scalar_value = value[0]
                                if isinstance(scalar_value, np.generic):
                                    scalar_value = scalar_value.item()
                                processed_params[key] = scalar_value
                            else:
                                processed_params[key] = str(value)
                        elif isinstance(value, np.generic):
                            # NumPy scalar types
                            processed_params[key] = value.item()
                        else:
                            processed_params[key] = value
                    row.update(processed_params)
                    data.append(row)

        df = pd.DataFrame(data)

        # Inspect DataFrame
        if self.enable_logging:
            print("DataFrame columns:", df.columns)
            print(f"Number of rows: {len(df)}")

        if 'Prompt Variant' in df.columns:
            if self.enable_logging:
                print(f"Number of unique prompt variants: {df['Prompt Variant'].nunique()}")
                print("\nSample of unique prompt variants:")
                print(df["Prompt Variant"].unique()[:5])
        else:
            if self.enable_logging:
                print("'Prompt Variant' column not found in DataFrame")

        if max_rows is not None:
            df = df.head(max_rows)

        # Separate columns
        param_columns = [col for col in df.columns if col not in ["Run Score", "Prompt Variant", "Full Prompt", "Generated Answer", "Ground Truth", "Query"]]

        if not param_columns:
            print("No parameter columns available for creating the heatmap.")
            return df

        # Ensure parameter columns have scalar values
        for param in param_columns:
            # Convert parameter columns to Python scalar types
            def convert_to_scalar(x):
                if isinstance(x, (list, np.ndarray)):
                    if len(x) == 1:
                        x = x[0]
                    else:
                        x = str(x)
                if isinstance(x, np.generic):
                    return x.item()
                else:
                    return x
            df[param] = df[param].apply(convert_to_scalar)

        # Now convert numeric columns to appropriate types
        for param in param_columns:
            # Try to convert to numeric
            try:
                df[param] = pd.to_numeric(df[param])
            except ValueError:
                pass  # Ignore if cannot convert to numeric

        # Create widgets
        param1_options = param_columns.copy()
        param2_options = param_columns.copy()

        # Set default parameters for initial heatmap
        default_param1 = param1_options[0]
        default_param2 = param2_options[1] if len(param2_options) > 1 else param1_options[0]

        param1 = widgets.Dropdown(options=param1_options, description='Parameter 1:', value=default_param1)
        param2 = widgets.Dropdown(options=param2_options, description='Parameter 2:', value=default_param2)
        score_metric = widgets.Dropdown(options=['Run Score'], description='Score Metric:')
        prompt_variant = widgets.Dropdown(options=['All'] + list(df['Prompt Variant'].unique()), description='Prompt Variant:')
        query = widgets.Dropdown(options=['All'] + list(df['Query'].unique()), description='Query:')
        heatmap_type = widgets.Dropdown(options=['Standard', 'Diverging', 'Categorical'], description='Heatmap Type:')

        output = widgets.Output()

        def create_heatmap(param1, param2, score, prompt, query_val, hmap_type):
            with output:
                output.clear_output(wait=True)

                # Filter data based on selections
                filtered_df = df.copy()
                if prompt != 'All':
                    filtered_df = filtered_df[filtered_df['Prompt Variant'] == prompt]
                if query_val != 'All':
                    filtered_df = filtered_df[filtered_df['Query'] == query_val]

                # Ensure that param1 and param2 columns are suitable for indexing
                if param1 not in filtered_df.columns or param2 not in filtered_df.columns:
                    print(f"Parameters '{param1}' or '{param2}' not found in data.")
                    return

                # Check for sufficient unique values
                if filtered_df[param1].nunique() < 2 or filtered_df[param2].nunique() < 2:
                    print(f"Not enough unique values in '{param1}' or '{param2}' to create a heatmap.")
                    print(f"Parameter 1 ({param1}) unique values: {filtered_df[param1].unique()}")
                    print(f"Parameter 2 ({param2}) unique values: {filtered_df[param2].unique()}")
                    print(f"\nConsider choosing different parameters or check the data type of '{param1}' and '{param2}'.")
                    return

                try:
                    # Create pivot table
                    pivot_df = filtered_df.pivot_table(values=score, index=param2, columns=param1, aggfunc='mean')

                    # Sort the pivot table index and columns if possible
                    try:
                        pivot_df = pivot_df.sort_index().sort_index(axis=1)
                    except Exception:
                        pass  # If sorting fails, ignore

                    # Create heatmap
                    plt.figure(figsize=(12, 10))
                    if hmap_type == 'Standard':
                        sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu")
                    elif hmap_type == 'Diverging':
                        sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0)
                    else:  # Categorical
                        sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="Set3")

                    plt.title(f"{score} for {param1} vs {param2}")
                    plt.tight_layout()
                    plt.show()

                    # Display summary statistics
                    print(f"\nSummary Statistics for {score}:")
                    print(filtered_df[score].describe())

                    # Display top 5 parameter combinations
                    print("\nTop 5 Parameter Combinations:")
                    top_5 = filtered_df.nlargest(5, score)
                    display_columns = [param1, param2, score, 'Prompt Variant', 'Query']
                    display_columns = [col for col in display_columns if col in filtered_df.columns]
                    print(top_5[display_columns].to_string(index=False))

                except Exception as e:
                    print(f"Error creating heatmap: {str(e)}")
                    print("\nThis error might occur when trying to create a heatmap with non-numeric or multi-dimensional data.")
                    print(f"Parameter 1 ({param1}) unique values: {filtered_df[param1].unique()}")
                    print(f"Parameter 2 ({param2}) unique values: {filtered_df[param2].unique()}")
                    print(f"\nConsider choosing different parameters or check the data type of '{param1}' and '{param2}'.")

        # Create interactive output
        try:
            from ipywidgets import interactive_output
        except ModuleNotFoundError:
            print("Please run `pip install ...` and run in a .ipynb to view visualizations & heatmaps")
            return
        controls = widgets.VBox([param1, param2, score_metric, prompt_variant, query, heatmap_type])
        out = interactive_output(create_heatmap, {
            'param1': param1,
            'param2': param2,
            'score': score_metric,
            'prompt': prompt_variant,
            'query_val': query,
            'hmap_type': heatmap_type
        })

        # Display widgets and initial heatmap
        display(controls, output)
        create_heatmap(param1.value, param2.value, score_metric.value, prompt_variant.value, query.value, heatmap_type.value)

        return df




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

    def create_run_results_table(
        self,
        experiment_result,
        max_rows=None,
        max_prompt_length=100,
        max_answer_length=500,
    ):
        try:
            import ipywidgets as widgets
            from IPython.display import display, HTML
        except ModuleNotFoundError:
            print("Please run `pip install ipywidgets` and `pip install IPython` and run in a .ipynb to view visualizations & heatmaps")
            return

        data = []
        for run_result in experiment_result.run_results:
            individual_results = run_result.metadata.get('Individual Simulation Results', [])
            for result in individual_results:
                # Extract data from the result
                answers = result.get('Answers', [])
                ground_truths = result.get('Ground Truth', [])
                prompt_variants = result.get('Prompt Variants', [])
                queries = result.get('Queries', [])

                if not all([answers, ground_truths, prompt_variants, queries]):
                    print(f"Missing or empty data in result: {result}")
                    continue

                # Ensure all lists have the same length
                min_length = min(len(answers), len(ground_truths), len(queries))

                # Create a row for each prompt variant and corresponding data
                for i in range(min_length):
                    for variant in set(prompt_variants):  # Use set to get unique variants
                        row = {
                            "Run Score": float(run_result.score),
                            "Prompt Variant": self._wrap_text(variant, max_prompt_length),
                            "Generated Answer": self._wrap_text(answers[i], max_answer_length),
                            "Ground Truth": self._wrap_text(ground_truths[i], max_answer_length),
                            "Query": self._wrap_text(queries[i], max_prompt_length),
                        }
                        row.update(run_result.params)
                        data.append(row)

        df = pd.DataFrame(data)

        # Inspect DataFrame
        print("DataFrame columns:", df.columns)
        print(f"Number of rows: {len(df)}")

        if 'Prompt Variant' in df.columns:
            print(f"Number of unique prompt variants: {df['Prompt Variant'].nunique()}")
            print("\nSample of unique prompt variants:")
            print(df["Prompt Variant"].unique()[:5])
        else:
            print("'Prompt Variant' column not found in DataFrame")

        if max_rows is not None:
            df = df.head(max_rows)

        # Create interactive elements
        param_columns = [col for col in df.columns if col not in ["Run Score", "Prompt Variant", "Generated Answer", "Ground Truth", "Query"]]

        dropdowns = {}
        for param in param_columns:
            options = ['All'] + list(df[param].unique())
            dropdowns[param] = widgets.Dropdown(options=options, description=param, style={'description_width': 'initial'})

        # Add Query dropdown
        query_options = ['All'] + list(df['Query'].unique())
        query_dropdown = widgets.Dropdown(options=query_options, description='Query', style={'description_width': 'initial'})

        # Add Prompt Variant dropdown
        prompt_variant_options = ['All'] + list(df['Prompt Variant'].unique())
        prompt_variant_dropdown = widgets.Dropdown(options=prompt_variant_options, description='Prompt Variant', style={'description_width': 'initial'})

        # Function to update table and recalculate score
        def update_table(*args):
            filtered_df = df.copy()
            for param, dropdown in dropdowns.items():
                if dropdown.value != 'All':
                    filtered_df = filtered_df[filtered_df[param] == dropdown.value]

            if query_dropdown.value != 'All':
                filtered_df = filtered_df[filtered_df['Query'] == query_dropdown.value]

            if prompt_variant_dropdown.value != 'All':
                filtered_df = filtered_df[filtered_df['Prompt Variant'] == prompt_variant_dropdown.value]

            # Calculate new score based on the filtered data
            if not filtered_df.empty:
                score = (filtered_df['Generated Answer'] == filtered_df['Ground Truth']).mean() * 100
            else:
                score = 0

            print(f"Filtered Score: {score:.2f}% accuracy based on applied filters")

            # Display the filtered DataFrame
            display(HTML(filtered_df.to_html(index=False)))

        for dropdown in dropdowns.values():
            dropdown.observe(update_table, names='value')
        query_dropdown.observe(update_table, names='value')
        prompt_variant_dropdown.observe(update_table, names='value')

        controls = widgets.VBox([widgets.Label("Select parameter combinations:")]
                                + list(dropdowns.values())
                                + [widgets.Label("Select Query:"), query_dropdown]
                                + [widgets.Label("Select Prompt Variant:"), prompt_variant_dropdown]
                                + [widgets.Button(description="Update Table")])

        output = widgets.Output()

        def on_button_click(b):
            with output:
                output.clear_output()
                update_table()

        controls.children[-1].on_click(on_button_click)

        display(controls, output)

        return df
