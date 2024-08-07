from datetime import datetime
from enum import Enum
from pathlib import Path
import traceback
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator

from llama_index.core.evaluation import BaseEvaluator
from llama_index.core.llms import CompletionResponse
from llama_index.core.base.response.schema import Response

from nomadic.model import OpenAIModel, SagemakerModel
from nomadic.result import RunResult, TunedResult
from nomadic.tuner.base import BaseParamTuner
from nomadic.util import is_ray_installed

from nomadic.experiment.prompt_tuning import PromptTuner, custom_evaluate

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


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

    param_dict: Dict[str, Any] = Field(
        ..., description="A dictionary of parameters to iterate over."
    )
    evaluation_dataset: List[Dict] = Field(
        default_factory=List,
        description="Evaluation dataset in dictionary format.",
    )
    user_prompt_request: str = Field(
        default="",
        description="User request for GPT prompt.",
    )
    model: Optional[Any] = Field(default=None, description="Model to run experiment")
    evaluator: Optional[Union[BaseEvaluator, Callable, Dict[str, Any]]] = Field(
        default=None,
        description="Evaluator of experiment (BaseEvaluator instance, callable, or dictionary)",
    )
    tuner: Optional[Any] = Field(default=None, description="Placeholder for base tuner")
    prompts: Optional[List[str]] = Field(
        default=None,
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
    tuned_result: Optional[TunedResult] = Field(
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
    num_example_prompts: Optional[int] = Field(
        default=0,
        description="Number of example prompts to include for few-shot prompting.",
    )

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

    # The get_fewshot_prompt_template method has been removed as it's no longer needed.
    # Prompt generation is now handled by the PromptTuner class in prompt_tuning.py.

    def run(self) -> TunedResult:
        is_error = False

        def get_responses(type_safe_param_values):
            all_pred_responses, all_eval_qs, all_ref_responses = [], [], []

            # Unpack the tuple returned by _enforce_param_types
            openai_params, prompt_tuning_params = type_safe_param_values

            if self.prompts:
                prompt_variants = self.prompts
                if self.enable_logging:
                    print(f"Using {len(prompt_variants)} provided prompts.")
            elif self.prompt_tuner is None:
                prompt_variants = [self.user_prompt_request]
                if self.enable_logging:
                    print("Prompt tuner is None. Using baseline prompt.")
            else:
                self.prompt_tuner.update_params(prompt_tuning_params)
                prompt_variants = self.prompt_tuner.generate_prompt_variants(
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
                    print(f"Prompt: {prompt_variant[:100]}...")

                pred_responses, eval_qs, ref_responses = [], [], []
                if not self.evaluation_dataset:
                    completion_response: CompletionResponse = self.model.run(
                        prompt=prompt_variant,
                        parameters=openai_params,  # Use openai_params here
                    )
                    pred_response = self._extract_response(completion_response)
                    pred_responses.append(pred_response)
                    eval_qs.append(prompt_variant)
                    ref_responses.append(None)
                    if self.enable_logging:
                        print(f"Response: {pred_response[:100]}...")
                else:
                    for j, example in enumerate(self.evaluation_dataset):
                        if self.enable_logging:
                            print(
                                f"Processing example {j+1}/{len(self.evaluation_dataset)}"
                            )
                        prompt = self._construct_prompt(prompt_variant, example)
                        completion_response: CompletionResponse = self.model.run(
                            prompt=prompt,
                            parameters=openai_params,  # Use openai_params here
                        )
                        pred_response = self._extract_response(completion_response)
                        pred_responses.append(pred_response)
                        eval_qs.append(prompt)
                        ref_responses.append(example.get("Answer", None))
                        if self.enable_logging:
                            print(f"Response: {pred_response[:100]}...")
                all_pred_responses.extend(pred_responses)
                all_eval_qs.extend(eval_qs)
                all_ref_responses.extend(ref_responses)
            return (all_pred_responses, all_eval_qs, all_ref_responses)

        def default_param_function(param_values: Dict[str, Any]) -> RunResult:
            if self.enable_logging:
                print("\nStarting new experiment run with parameters:")
                for param, value in param_values.items():
                    print(f"{param}: {value}")

            type_safe_param_values = self._enforce_param_types(param_values)
            pred_responses, eval_qs, ref_responses = get_responses(
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
            }

            if self.prompt_tuner:
                metadata["Prompting Approaches"] = self.prompt_tuner.approach
                metadata["Prompt Complexities"] = self.prompt_tuner.complexity
                metadata["Prompt Focuses"] = self.prompt_tuner.focus
            if hasattr(self, "evaluation_metrics"):
                metadata["Evaluation Metrics"] = self.evaluation_metrics

            if eval_qs:
                metadata["Prompts"] = eval_qs

            return RunResult(
                score=mean_score,
                params=param_values,
                metadata=metadata,
            )

        self.experiment_status = ExperimentStatus("running")
        self.start_datetime = datetime.now()
        result = None
        try:
            if self.enable_logging:
                print("\nSetting up tuner...")
            self._setup_tuner(default_param_function)
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
        self.tuned_result = result or self._create_default_tuned_result()
        if self.enable_logging:
            print(f"\nExperiment completed. Status: {self.experiment_status}")
        return self.tuned_result

    def _construct_prompt(self, prompt_variant: str, example: Dict[str, str]) -> str:
        prompt = f"{prompt_variant}\n\n"
        if self.num_example_prompts > 0:
            prompt += self._add_few_shot_examples()
        prompt += f"Context: {example['Context']}\n\n"
        prompt += f"Instruction: {example['Instruction']}\n\n"
        prompt += f"Question: {example['Question']}\n\n"
        return prompt

    def _add_few_shot_examples(self) -> str:
        few_shot_examples = ""
        for i in range(min(self.num_example_prompts, len(self.evaluation_dataset) - 1)):
            example = self.evaluation_dataset[i]
            few_shot_examples += f"Example {i+1}:\n"
            few_shot_examples += f"Context: {example['Context']}\n"
            few_shot_examples += f"Instruction: {example['Instruction']}\n"
            few_shot_examples += f"Question: {example['Question']}\n"
            few_shot_examples += f"Answer: {example['Answer']}\n\n"
        return few_shot_examples

    # This method has been removed as it's no longer needed

    def _extract_response(self, completion_response: CompletionResponse) -> str:
        if isinstance(self.model, OpenAIModel):
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
                        # Get the OpenAI API key from the model
                        openai_api_key = (
                            self.model.api_keys.get("OPENAI_API_KEY")
                            if hasattr(self.model, "api_keys")
                            else None
                        )
                        # Pass the OpenAI API key to custom_evaluate
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

        return sum(scores) / len(scores) if scores else 0

    def _setup_tuner(self, param_function: Callable):
        if not self.tuner:
            if self.enable_logging:
                print("\nSetting up tuner...")
            if is_ray_installed():
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
                # FlamlParamTuner setup remains unchanged
                from nomadic.tuner import FlamlParamTuner

                self.tuner = FlamlParamTuner(
                    param_fn=param_function,
                    param_dict=self.param_dict,
                    search_method=self.search_method,
                    fixed_param_dict=self.fixed_param_dict,
                    current_param_dict=self.current_param_dict,
                    show_progress=self.enable_logging,
                )

    def _format_error_message(self, exception: Exception) -> str:
        return f"Exception: {str(exception)}\n\nStack Trace:\n{traceback.format_exc()}"

    def _determine_experiment_status(self, is_error: bool) -> ExperimentStatus:
        return (
            ExperimentStatus("finished_success")
            if not is_error
            else ExperimentStatus("finished_error")
        )

    def _create_default_tuned_result(self) -> TunedResult:
        return TunedResult(
            run_results=[RunResult(score=-1, params={}, metadata={})],
            best_idx=0,
        )

    def save_experiment(self, folder_path: Path):
        file_name = (
            f"/experiment_{self.start_datetime.strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )
        with open(folder_path + file_name, "w") as file:
            file.write(self.model_dump_json(exclude=("model", "evaluator")))

    def visualize_results(self):
        if not self.tuned_result:
            if self.enable_logging:
                print("No results to visualize.")
            return

        # Extract scores and parameters from run_results
        scores = [run.score for run in self.tuned_result.run_results]
        params = [run.params for run in self.tuned_result.run_results]

        # Create a DataFrame
        df = pd.DataFrame(params)
        df["score"] = scores

        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns

        # Visualize score distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x="score", kde=True)
        plt.title("Distribution of Scores")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.show()

        # Boxplots for categorical parameters vs score
        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=col, y="score", data=df)
            plt.title(f"Score Distribution by {col}")
            plt.ylabel("Score")
            plt.xticks(rotation=45)
            plt.show()

        # Scatterplots for numeric parameters vs score
        for col in numeric_cols:
            if col != "score":
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=col, y="score", data=df)
                plt.title(f"Score vs {col}")
                plt.ylabel("Score")
                plt.show()

        # Summary statistics of scores
        print("Score Summary Statistics:")
        print(df["score"].describe())

        # Top 5 performing parameter combinations
        print("\nTop 5 Performing Parameter Combinations:")
        top_5 = df.sort_values("score", ascending=False).head()
        print(top_5.to_string(index=False))
