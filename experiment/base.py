from abc import abstractmethod
from datetime import datetime
from enum import Enum
import traceback
from typing import Any, Dict, Iterable, List, Optional
from pydantic import BaseModel, Field

import numpy as np
from llama_index.core.evaluation import BatchEvalRunner
from llama_index.core.llms import CompletionResponse
from llama_index.core.base.response.schema import Response

from nomadic.model import Model
from nomadic.result import RunResult, TunedResult
from nomadic.tuner import BaseParamTuner, RayTuneParamTuner

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

    # Required
    model: Model = Field(..., description="Model to run experiment")
    hp_space: Dict[str, Any] = Field(
        ..., description="A dictionary of parameters to iterate over."
    )
    evaluator: BatchEvalRunner = Field(
        ..., description="Evaluator of experiment"
    )
    evaluation_dataset: Dict = Field(
        default_factory=dict,
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
            for row in self.evaluation_dataset["data"]:
                completion_response: CompletionResponse = (
                    self.model.llm.complete(
                        prompt=row["Instruction"],
                        kwargs={"parameters": param_values},
                    )
                )
                pred_responses.append(completion_response.text)
                eval_qs.append(row["Instruction"])
                ref_responses.append(row.get("Answer", None))

            # TODO: Generalize
            eval_results = self.eval_batch_runner.evaluate_responses(
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

        try:
            tuner = RayTuneParamTuner(
                param_fn=default_param_function,
                param_dict=self.hp_space,
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
        return result
