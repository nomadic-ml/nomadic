from datetime import datetime
from enum import Enum
import os
from pathlib import Path
import traceback
from typing import Any, Callable, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from llama_index.core.evaluation import BaseEvaluator
from llama_index.core.llms import CompletionResponse
from llama_index.core.base.response.schema import Response

from nomadic.model import OpenAIModel, SagemakerModel
from nomadic.result import RunResult, TunedResult
from nomadic.tuner import ParamTuner, RayTuneParamTuner, BaseParamTuner

"""
experiment = {
    experiment_runs = {
        experiment_run: {
            selected_hp_values = {
                'hp_name': HP_VALUE: Iterable
            },
        },
    },
    current_hp_values = {
        'hp_name': HP_VALUE: Iterable
    },
    hp_search_space_map = {
        'hp_name': hp_search_space
    },
    datetime_started = datetime,
    datetime_ended = datetime,
    author = User
}
"""


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
    """Base experiment run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Required
    param_dict: Dict[str, Dict[str, Any]] = Field(
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
    # TODO: Figure out why Union[SagemakerModel, OpenAIModel] doesn't work
    # Note: A model is always required. It is currently denoted as `Optional` brcause of the TODO above.
    model: Optional[Any] = Field(default=None, description="Model to run experiment")
    evaluator: Optional[BaseEvaluator] = Field(
        default=None, description="Evaluator of experiment"
    )
    # TODO: Add BaseParamTuner (or ParamTuner, RayTuneParamTuner) as proper type here
    tuner: Optional[Any] = Field(default=None, description="Placeholder for base tuner")
    # Optional
    fixed_param_dict: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional dictionary of fixed hyperparameter values.",
    )
    current_param_dict: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional dictionary of current hyperparameter values.",
    )
    num_prompts: int = Field(
        default=1,
        description="Number of prompt variations to generate for each data point.",
    )
    search_method: Optional[str] = Field(
        default="grid", description="Tuner search option. Can be: [grid, bayesian]"
    )
    # Self populated
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

    def model_post_init(self, ctx):
        if self.search_method not in ("grid", "bayesian"):
            raise ValueError(
                f"Selected Experiment search_method `{self.search_method}` is not valid."
            )
        self.tuner = None

    def generate_similar_prompts(self, prompt: str, user_request: str) -> List[str]:
        """
        Generate similar prompts using a GPT query.
        """
        from openai import OpenAI

        client = OpenAI(api_key=self.model.api_keys["OPENAI_API_KEY"])

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"Create similar prompts to the one given: {prompt}. User request: {user_request}",
                }
            ],
            max_tokens=100,
            n=self.num_prompts,
            stop=None,
            temperature=0.7,
        )

        similar_prompts = [
            choice.message.content.strip() for choice in response.choices
        ]
        return similar_prompts

    def run(self) -> TunedResult:
        """Run experiment."""
        is_error = False

        def default_param_function(param_values: Dict[str, Any]) -> RunResult:
            contexts, pred_responses, eval_qs, ref_responses = [], [], [], []
            # Enforce param values to fit their default types, if they exist.
            type_safe_param_values = {}
            for param, val in param_values.items():
                if param in self.model.hyperparameters.default:
                    type_safe_param_values[param] = self.model.hyperparameters.default[
                        param
                    ]["type"](val)
                else:
                    type_safe_param_values[param] = val

            if not self.evaluation_dataset:
                completion_response: CompletionResponse = self.model.run(
                    context=None,
                    instruction=None,
                    parameters=type_safe_param_values,
                )
            for row in self.evaluation_dataset:
                similar_prompts = self.generate_similar_prompts(
                    row["Instruction"], self.user_prompt_request
                )
                for prompt in similar_prompts:
                    completion_response: CompletionResponse = self.model.run(
                        context=row["Context"],
                        instruction=prompt,
                        parameters=type_safe_param_values,
                    )
                    # OpenAI's model returns result in `completion_response.text`.
                    # Sagemaker's model returns result in `completion_response.raw["Body"]`.
                    if self.model:
                        if isinstance(self.model, OpenAIModel):
                            pred_response = completion_response.text
                        elif isinstance(self.model, SagemakerModel):
                            pred_response = completion_response.raw["Body"]
                        else:
                            raise NotImplementedError
                        contexts.append(row.get("Context", None))
                        pred_responses.append(pred_response)
                        eval_qs.append(prompt)
                        ref_responses.append(row.get("Answer", None))

            eval_results = []
            if self.evaluator:
                for i, response in enumerate(pred_responses):
                    eval_results.append(
                        self.evaluator.evaluate_response(
                            response=Response(response), reference=ref_responses[i]
                        )
                    )

            # TODO: Generalize
            # get semantic similarity metric
            scores = [r.score for r in eval_results]
            mean_score = sum(scores) / len(scores) if scores else 0
            return RunResult(
                score=mean_score,
                params=param_values,
                metadata={
                    "Contexts": contexts,
                    "Instructions": eval_qs,
                    "Predictions": pred_responses,
                    "Referance Responses": ref_responses,
                },
            )

        self.experiment_status = ExperimentStatus("running")
        self.start_datetime = datetime.now()
        result = None
        try:
            self.tuner = RayTuneParamTuner(
                param_fn=default_param_function,
                param_dict=self.param_dict,
                search_method=self.search_method,
                fixed_param_dict=self.fixed_param_dict,
                current_param_dict=self.current_param_dict,
                show_progress=True,
            )
            result = self.tuner.fit()
        except Exception as e:
            is_error = True
            self.experiment_status_message = (
                f"Exception: {str(e)}\n\nStack Trace:\n{traceback.format_exc()}"
            )

        self.end_datetime = datetime.now()
        self.experiment_status = (
            ExperimentStatus("finished_success")
            if not is_error
            else ExperimentStatus("finished_error")
        )
        self.tuned_result = result or TunedResult(
            run_results=[RunResult(score=-1, params={}, metadata={})],
            best_idx=0,
        )
        return self.tuned_result

    def save_experiment(self, folder_path: Path):
        file_name = f"/experiment_{self.start_datetime}.json"
        with open(folder_path + file_name, "w") as file:
            file.write(self.model_dump_json(exclude=("model", "evaluator")))
