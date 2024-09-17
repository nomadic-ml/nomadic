import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from nomadic.result import ExperimentResult, RunResult
from nomadic.experiment.base import Experiment
from nomadic.attack_models.tap_algorithm import TAPAlgorithm
from nomadic.attack_models.iris_algorithm import IRISAlgorithm
from nomadic.attack_models.rl_agent import RLAgent
from nomadic.attack_models.evaluators import GPTEvaluator
from nomadic.attack_models.language_models import GPT

class AISafetyExperiment(Experiment):
    def __init__(self, algorithm_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_attributes = {}
        self._algorithms = {
            'TAP': TAPAlgorithm(self.model, self.model, self.evaluator, width=3, branching_factor=2, depth=3),
            'IRIS': IRISAlgorithm(self.model, self.model, self.evaluator, max_refinement_steps=3),
            'RLAgent': RLAgent(self.model, self.model, self.evaluator)
        }
        self.set_custom_attribute('algorithm_type', algorithm_type)

    def get_custom_attribute(self, attr_name: str) -> Any:
        return self._custom_attributes.get(attr_name)

    def set_custom_attribute(self, attr_name: str, value: Any) -> None:
        self._custom_attributes[attr_name] = value

    @property
    def algorithm_type(self) -> str:
        return self.get_custom_attribute('algorithm_type')

    def run_attack(self, param_dict: Dict[str, Any]) -> Dict[str, Any]:
        param_dict['initial_prompt'] = self.user_prompt_request
        if self.algorithm_type in self._algorithms:
            return self._algorithms[self.algorithm_type].run_attack(**param_dict)
        else:
            raise ValueError(f"Unknown algorithm type: {self.algorithm_type}")

    def run(self, param_dict: Dict[str, Any]) -> ExperimentResult:
        result = self.run_attack(param_dict)
        result['algorithm_type'] = self.algorithm_type  # Include algorithm_type in the metadata
        return ExperimentResult(
            hp_search_space=param_dict,
            run_results=[RunResult(score=result['best_score'], params=param_dict, metadata=result)]
        )

    def visualize_results(self, results: List[ExperimentResult]):
        algorithms = [result.run_results[0].metadata['algorithm_type'] for result in results]
        scores = [result.run_results[0].score for result in results]

        plt.figure(figsize=(10, 6))
        plt.bar(algorithms, scores)
        plt.title('Algorithm Performance Comparison')
        plt.xlabel('Algorithm')
        plt.ylabel('Best Score')
        plt.ylim(0, 10)
        plt.savefig('algorithm_comparison.png')
        plt.close()
