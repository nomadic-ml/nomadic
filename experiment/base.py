from datetime import datetime
from enum import Enum
from pathlib import Path
import traceback
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

import numpy as np
from llama_index.core.evaluation import BatchEvalRunner
from llama_index.core.llms import CompletionResponse
from llama_index.core.base.response.schema import Response

from nomadic.model import OpenAIModel, SagemakerModel
from nomadic.result import RunResult, TunedResult
from nomadic.tuner import ParamTuner

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
    model: Optional[Any] = Field(
        default=None, description="Model to run experiment"
    )
    evaluator: Optional[BatchEvalRunner] = Field(
        default=None, description="Evaluator of experiment"
    )
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
    # Self populated
    start_datetime: Optional[datetime] = Field(
        default=None, description="Start datetime."
    )
    end_datetime: Optional[datetime] = Field(
        default=None, description="End datetime."
    )
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

    def generate_similar_prompts(self, prompt: str, user_request: str) -> List[str]:
        """
        Generate similar prompts using a GPT query.
        """
        import openai

        #openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key= "sk-proj-gSjHA2Ve0MwmGbo5KcPuT3BlbkFJwbGxbpmjK22mQmXNgwhZ"
        response = openai.Completion.create(
            engine="davinci-codex",
            prompt=f"Create similar prompts to the one given: {prompt}. User request: {user_request}",
            max_tokens=100,
            n=self.num_prompts,
            stop=None,
            temperature=0.7,
        )

        similar_prompts = [choice["text"].strip() for choice in response.choices]
        return similar_prompts

    def run(self) -> TunedResult:
        """Run experiment."""
        is_error = False

        def default_param_function(param_values: Dict[str, Any]) -> RunResult:
            contexts, pred_responses, eval_qs, ref_responses = [], [], [], []
            for row in self.evaluation_dataset:
                similar_prompts = self.generate_similar_prompts(row["Instruction"], self.user_prompt_request)
                for prompt in similar_prompts:
                    completion_response: CompletionResponse = self.model.run(
                        context=row["Context"],
                        instruction=prompt,
                        parameters=param_values,
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

            eval_results = self.evaluator.evaluate_responses(
                # eval_qs, responses=pred_responses, reference=ref_responses
                responses=[Response(response) for response in pred_responses],
                reference=ref_responses,
            )

            # TODO: Generalize
            # get semantic similarity metric
            mean_score = np.array(
                [r.score for r in eval_results["semantic_similarity"]]
            ).mean()
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
            # TODO: Figure out serialization problem caused by
            # an `_abc_data` object.
            # tuner = RayTuneParamTuner(
            #     param_fn=default_param_function,
            #     param_dict=self.hp_space,
            #     fixed_param_dict=self.fixed_param_dict,
            #     current_param_dict=self.current_param_dict,
            #     show_progress=True,
            # )
            tuner = ParamTuner(
                param_fn=default_param_function,
                param_dict=self.param_dict,
                search_method="grid",
                fixed_param_dict=self.fixed_param_dict,
                current_param_dict=self.current_param_dict,
                show_progress=True,
            )
            result = tuner.fit()
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
