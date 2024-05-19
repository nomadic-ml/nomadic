from abc import abstractmethod
import datetime
from typing import Callable, Dict, Iterable, Set
from pydantic import BaseModel, Field, ValidationError

"""
experiment_run = {
    experiments = {
        experiment: {
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
    """Base experiment"""

    selected_hp_values = Dict[str, Iterable] = Field(
        ..., description="Hyperparameter values of experiment to run."
    )

class ExperimentRun(BaseModel):
    """Base experiment run."""

    experiments: Set[Experiment] = Field(
        ..., description="Set of experiments."
    )
    start_datetime: datetime = Field(
        ..., description="Start datetime."
    )
    end_datetime: datetime = Field(
        ..., description="End datetime."
    )

    @abstractmethod
    def fit(self) -> TunedResult:
        """Tune parameters."""