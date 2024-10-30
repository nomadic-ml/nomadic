"""Default experiment implementation that supports both parameter-based and dataset-based experiments."""
from typing import Any, Dict
from .base_experiment import BaseExperiment
from nomadic.experiment.helpers import DefaultResultManager as ResultManager, DefaultExperimentSetup, DefaultResponseManager

class DefaultExperiment(BaseExperiment):
    """Default implementation of BaseExperiment supporting both experiment patterns."""

    def run(self, param_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Run the experiment with the given parameters."""
        try:
            # Validate and setup
            self._validate_and_setup(param_dict)

            # Run appropriate experiment type
            if self.evaluation_dataset:
                results = self._run_dataset_experiment(param_dict)
            else:
                results = self._run_parameter_experiment(param_dict)

            # Update status and save results
            results = ResultManager.update_status(results, self.status)

            if self.enable_logging:
                ResultManager.save_results(results, self.name)

            return results

        except Exception as e:
            raise e
