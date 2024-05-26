from datetime import datetime
from enum import Enum
import traceback
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, ConfigDict, Field

import numpy as np
from llama_index.core.evaluation import BatchEvalRunner
from llama_index.core.llms import CompletionResponse
from llama_index.core.base.response.schema import Response

from nomadic.model import Model
from nomadic.model.base import OpenAIModel, SagemakerModel
from nomadic.result import RunResult, TunedResult
from nomadic.tuner import RayTuneParamTuner, ParamTuner

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
    # TODO: Figure out why Union[SagemakerModel, OpenAIModel] doesn't work
    model: Any = Field(..., description="Model to run experiment")
    param_dict: Dict[str, Dict[str, Any]] = Field(
        ..., description="A dictionary of parameters to iterate over."
    )
    evaluator: BatchEvalRunner = Field(
        ..., description="Evaluator of experiment"
    )
    evaluation_dataset: List[Dict] = Field(
        default_factory=List,
        description="Evaluation dataset in dictionary format.",
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
    # Self populated
    start_datetime: Optional[datetime] = Field(
        default=None, description="Start datetime."
    )
    end_datetime: Optional[datetime] = Field(
        default=None, description="End datetime."
    )
    experiment_status: ExperimentStatus = Field(
        default=ExperimentStatus("not_started"),
        description="Current status of Experiment",
    )
    experiment_status_message: str = Field(
        default="",
        description="Detailed description of Experiment status during error.",
    )

    def run(self) -> TunedResult:
        """Run experiment."""
        is_error = False

        def default_param_function(param_values: Dict[str, Any]) -> RunResult:
            eval_qs, pred_responses, ref_responses = [], [], []
            for row in self.evaluation_dataset:
                completion_response: CompletionResponse = self.model.run(
                    context=row["Context"],
                    instruction=row["Instruction"],
                    parameters=param_values,
                )
                # OpenAI's model returns result in `completion_response.text`.
                # Sagemaker's model returns result in `completion_response.raw["Body"]`.
                if isinstance(self.model, OpenAIModel):
                    pred_response = completion_response.text
                elif isinstance(self.model, SagemakerModel):
                    pred_response = completion_response.raw["Body"]
                else:
                    raise NotImplementedError
                pred_responses.append(pred_response)
                eval_qs.append(row["Instruction"])
                ref_responses.append(row.get("Answer", None))

            # TODO: Generalize
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
            return RunResult(score=mean_score, params=param_values)

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
        return result or TunedResult(
            run_results=[RunResult(score=-1, params={}, metadata={})],
            best_idx=0,
        )
