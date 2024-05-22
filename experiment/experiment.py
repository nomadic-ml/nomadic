from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from pydantic import BaseModel, Field

from llama_index.core.llms import LLM
from llama_index.core.evaluation import BatchEvalRunner

from nomadic.model.model import Model
from nomadic.result.result import RunResult, TunedResult
from nomadic.tune.tuner import BaseParamTuner, RayTuneParamTuner

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


class Experiment(BaseModel):
    """Base experiment run."""

    evaluator: BatchEvalRunner = Field(
        ..., description="Evaluator of experiment"
    )
    tuner: BaseParamTuner = Field(..., description="Tuner of ")
    model: Model = Field(..., description="Model to run experiment")
    current_param_dict: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional dictionary of current hyperparameter values.",
    )
    evaluation_dataset: Dict = Field(
        default_factory=dict,
        description="Optional dictionary of current hyperparameter values.",
    )
    start_datetime: Optional[datetime] = Field(
        default=None, description="Start datetime."
    )
    end_datetime: Optional[datetime] = Field(
        default=None, description="End datetime."
    )

    def run(self) -> TunedResult:
        """Run experiment."""

        def objective_function(param_values: Dict[str, Any]) -> RunResult:
            self.model.llm.complete()

        self.start_datetime = datetime.now()

        tuner = RayTuneParamTuner()

        self.end_datetime = datetime.now()
