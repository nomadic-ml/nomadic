import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import io
from .base_visualization import BaseVisualization, ExperimentResult

class RAGVisualization(BaseVisualization):
    """Visualization class for RAG-specific experiment results."""

    def __init__(self, experiment_result: ExperimentResult):
        """Initialize RAG visualization with experiment results."""
        super().__init__(experiment_result)
        self.param_combinations = self._extract_param_combinations()
        self.accuracy_scores = self._extract_accuracy_scores()

    def _extract_param_combinations(self) -> List[Dict[str, Any]]:
        """Extract parameter combinations from experiment results."""
        return [run.metadata.get("Prompt Parameters", {})
                for run in self.experiment_result.run_results]

    def _extract_accuracy_scores(self) -> List[float]:
        """Extract accuracy scores from experiment results."""
        return [run.metadata.get("accuracy_score", 0.0)
                for run in self.experiment_result.run_results]

    def create_retrieval_heatmap(self) -> Optional[bytes]:
        """
        Create a heatmap visualization for retrieval parameter combinations and their accuracy scores.

        Returns:
            bytes: The visualization as a bytes object
        """
        # Convert parameters and scores into a format suitable for heatmap
        unique_params = {}
        for params in self.param_combinations:
            for key, value in params.items():
                if key not in unique_params:
                    unique_params[key] = set()
                unique_params[key].add(str(value))

        # Select two parameters with the most variation for visualization
        plot_params = sorted(unique_params.items(), key=lambda x: len(x[1]), reverse=True)[:2]
        if len(plot_params) < 2:
            return None

        param1, param2 = plot_params[0][0], plot_params[1][0]

        # Create matrix for heatmap
        param1_values = sorted(list(unique_params[param1]))
        param2_values = sorted(list(unique_params[param2]))
        heatmap_data = np.zeros((len(param1_values), len(param2_values)))
        count_matrix = np.zeros((len(param1_values), len(param2_values)))

        for params, score in zip(self.param_combinations, self.accuracy_scores):
            i = param1_values.index(str(params[param1]))
            j = param2_values.index(str(params[param2]))
            heatmap_data[i, j] += score
            count_matrix[i, j] += 1

        # Average scores where there are multiple entries
        mask = count_matrix > 0
        heatmap_data[mask] /= count_matrix[mask]

        # Create visualization
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data,
                    annot=True,
                    fmt='.3f',
                    xticklabels=param2_values,
                    yticklabels=param1_values,
                    cmap='viridis')

        plt.title(f'Retrieval Performance Heatmap\n{param1} vs {param2}')
        plt.xlabel(param2)
        plt.ylabel(param1)

        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        return buf.getvalue()

    def create_inference_heatmap(self) -> Optional[bytes]:
        """
        Create a heatmap visualization for inference parameter combinations and their accuracy scores.

        Returns:
            bytes: The visualization as a bytes object
        """
        # Similar structure to create_retrieval_heatmap but for inference parameters
        unique_params = {}
        for params in self.param_combinations:
            for key, value in params.items():
                if key not in unique_params:
                    unique_params[key] = set()
                unique_params[key].add(str(value))

        plot_params = sorted(unique_params.items(), key=lambda x: len(x[1]), reverse=True)[:2]
        if len(plot_params) < 2:
            return None

        param1, param2 = plot_params[0][0], plot_params[1][0]

        param1_values = sorted(list(unique_params[param1]))
        param2_values = sorted(list(unique_params[param2]))
        heatmap_data = np.zeros((len(param1_values), len(param2_values)))
        count_matrix = np.zeros((len(param1_values), len(param2_values)))

        for params, score in zip(self.param_combinations, self.accuracy_scores):
            i = param1_values.index(str(params[param1]))
            j = param2_values.index(str(params[param2]))
            heatmap_data[i, j] += score
            count_matrix[i, j] += 1

        mask = count_matrix > 0
        heatmap_data[mask] /= count_matrix[mask]

        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data,
                    annot=True,
                    fmt='.3f',
                    xticklabels=param2_values,
                    yticklabels=param1_values,
                    cmap='viridis')

        plt.title(f'Inference Performance Heatmap\n{param1} vs {param2}')
        plt.xlabel(param2)
        plt.ylabel(param1)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        return buf.getvalue()

    def create_interactive_visualization(self) -> None:
        """Create interactive visualizations specific to RAG experiments."""
        self.visualize_rag_results()

    def visualize_rag_results(self, title: str = "RAG Results Visualization") -> bytes:
        """
        Create a comprehensive visualization of RAG results including accuracy scores and explanations.

        Args:
            title: Title for the visualization

        Returns:
            bytes: The visualization as a bytes object
        """
        explanations = [result.metadata.get('explanation', '')
                       for result in self.experiment_result.run_results]

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 2])

        # Plot accuracy scores
        ax1.plot(self.accuracy_scores, marker='o')
        ax1.set_title('Accuracy Scores Over Runs')
        ax1.set_xlabel('Run Number')
        ax1.set_ylabel('Accuracy Score')
        ax1.grid(True)

        # Plot explanation lengths/complexity
        explanation_lengths = [len(exp) for exp in explanations]
        ax2.bar(range(len(explanation_lengths)), explanation_lengths)
        ax2.set_title('Explanation Lengths by Run')
        ax2.set_xlabel('Run Number')
        ax2.set_ylabel('Explanation Length')

        plt.suptitle(title)
        plt.tight_layout()

        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        return buf.getvalue()
