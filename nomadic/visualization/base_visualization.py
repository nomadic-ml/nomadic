import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass
from ipywidgets import interact, widgets

@dataclass
class ExperimentResult:
    """Data class for experiment results."""
    run_results: List[Any]
    metadata: Dict[str, Any]

class BaseVisualization(ABC):
    """Base class for experiment result visualization."""

    def __init__(self, experiment_result: ExperimentResult):
        """Initialize visualization with experiment results."""
        self.experiment_result = experiment_result
        self.data = self._extract_data_from_results()
        self.df = self._create_and_clean_dataframe()
        self.numeric_cols, self.categorical_cols = self._separate_column_types()

    def visualize_results(self, graphs_to_include: Optional[List[str]] = None) -> None:
        """
        Visualize experiment results with various plots and graphs.

        Args:
            graphs_to_include: List of graph types to include in visualization
        """
        if not self.experiment_result or not self.experiment_result.run_results:
            print("No results to visualize.")
            return

        if graphs_to_include is None:
            graphs_to_include = [
                "overall_score_distribution",
                "metric_score_distributions",
                "parameter_relationships",
                "summary_statistics",
                "correlation_heatmap",
            ]

        self._generate_visualizations(graphs_to_include)

    def _generate_visualizations(self, graphs_to_include: List[str]) -> None:
        """Generate requested visualizations."""
        if "overall_score_distribution" in graphs_to_include:
            self._plot_score_distribution("overall_score", "Distribution of Overall Scores")

        if "metric_score_distributions" in graphs_to_include:
            for metric in self.data["all_metric_scores"].keys():
                if metric in self.df.columns:
                    self._plot_score_distribution(metric, f"Distribution of {metric} Scores")

        if "parameter_relationships" in graphs_to_include:
            self._visualize_parameter_relationships()

        if "summary_statistics" in graphs_to_include:
            self._print_summary_statistics()

        if "correlation_heatmap" in graphs_to_include:
            self._create_correlation_heatmap()

    def _extract_data_from_results(self) -> Dict[str, Any]:
        """Extract data from experiment results."""
        all_scores = []
        all_params = []
        all_metric_scores = {}

        for run in self.experiment_result.run_results:
            metadata = run.metadata
            all_scores.append(run.score)
            all_params.append(metadata.get("Prompt Parameters", {}))

            eval_result = metadata.get("Custom Evaluator Results", [{}])[0]
            if "scores" in eval_result:
                for metric, score in eval_result["scores"].items():
                    all_metric_scores.setdefault(metric, []).append(score)

        return {
            "all_scores": all_scores,
            "all_params": all_params,
            "all_metric_scores": all_metric_scores,
        }

    def _create_and_clean_dataframe(self) -> pd.DataFrame:
        """Create and clean DataFrame from extracted data."""
        df = pd.DataFrame(self.data["all_params"])
        df["overall_score"] = self.data["all_scores"]

        for metric, scores in self.data["all_metric_scores"].items():
            df[metric] = scores

        return df.dropna(axis=1, how="any").dropna()

    def _separate_column_types(self) -> tuple[pd.Index, pd.Index]:
        """Separate numeric and categorical columns."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns
        return numeric_cols, categorical_cols

    def _plot_score_distribution(self, column: str, title: str) -> None:
        """Plot score distribution for a given column."""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x=column, kde=True)
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

    def _visualize_parameter_relationships(self) -> None:
        """Visualize relationships between parameters and scores."""
        for col in self.categorical_cols:
            if col != "overall_score":
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=col, y="overall_score", data=self.df)
                plt.title(f"Overall Score Distribution by {col}")
                plt.ylabel("Overall Score")
                plt.xticks(rotation=45)
                plt.show()

        for col in self.numeric_cols:
            if col != "overall_score":
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=col, y="overall_score", data=self.df)
                plt.title(f"Overall Score vs {col}")
                plt.ylabel("Overall Score")
                plt.show()

    def _print_summary_statistics(self) -> None:
        """Print summary statistics for scores."""
        print("Score Summary Statistics:")
        print(self.df[["overall_score"] + list(self.data["all_metric_scores"].keys())].describe())

        print("\nTop 5 Performing Parameter Combinations:")
        top_5 = self.df.sort_values("overall_score", ascending=False).head()
        print(top_5.to_string(index=False))

    def _create_correlation_heatmap(self) -> None:
        """Create correlation heatmap for numeric parameters."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.df[self.numeric_cols].corr(), annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap of Numeric Parameters and Scores")
        plt.show()

    @abstractmethod
    def create_interactive_visualization(self) -> None:
        """Create interactive visualizations (to be implemented by subclasses)."""
        pass
