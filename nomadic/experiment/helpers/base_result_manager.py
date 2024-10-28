from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
from datetime import datetime
from nomadic.result import ExperimentResult
from pathlib import Path
import json
import traceback
from pydantic import BaseModel

class BaseResultManager(ABC):
    """Base class for managing experiment results."""

    @abstractmethod
    def format_error_message(self, exception: Exception) -> str:
        """Format an error message from an exception."""
        pass

    @abstractmethod
    def determine_experiment_status(self, is_error: bool) -> str:
        """Determine the experiment status based on error state."""
        pass

    @abstractmethod
    def create_default_experiment_result(self, param_dict: Dict[str, Any]) -> ExperimentResult:
        """Create a default experiment result when the experiment fails."""
        pass

    @abstractmethod
    def save_experiment(self, folder_path: str, start_datetime: datetime, experiment_data: Any) -> None:
        """Save the experiment data to a file."""
        pass

class DefaultResultManager(BaseResultManager, BaseModel):
    """Default implementation of result management functionality."""

    def format_error_message(self, exception: Exception) -> str:
        """Format an error message from an exception with stack trace."""
        error_message = f"Error: {str(exception)}\n"
        error_message += "Stack trace:\n"
        error_message += traceback.format_exc()
        return error_message

    def determine_experiment_status(self, is_error: bool) -> str:
        """Determine experiment status based on error state."""
        return "ERROR" if is_error else "SUCCESS"

    def create_default_experiment_result(self, param_dict: Dict[str, Any]) -> ExperimentResult:
        """Create a default experiment result with error status."""
        return ExperimentResult(
            parameters=param_dict,
            status="ERROR",
            metrics={
                "error_rate": 1.0,
                "success_rate": 0.0
            },
            metadata={
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            }
        )

    def save_experiment(self, folder_path: str, start_datetime: datetime, experiment_data: Any) -> None:
        """Save experiment data to a timestamped JSON file."""
        # Create folder if it doesn't exist
        save_path = Path(folder_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Create timestamped filename
        timestamp = start_datetime.strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_results_{timestamp}.json"
        file_path = save_path / filename

        # Convert experiment data to JSON-serializable format
        if hasattr(experiment_data, "dict"):
            data = experiment_data.dict()
        elif hasattr(experiment_data, "__dict__"):
            data = experiment_data.__dict__
        else:
            data = experiment_data

        # Save to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
