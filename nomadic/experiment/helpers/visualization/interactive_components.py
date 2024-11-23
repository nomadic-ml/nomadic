import ipywidgets as widgets
from IPython.display import display, HTML
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .base_visualization import BaseVisualization, ExperimentResult

class InteractiveVisualization(BaseVisualization):
    """Class for creating interactive visualizations of experiment results."""

    def __init__(self, experiment_result: ExperimentResult):
        """Initialize interactive visualization with experiment results."""
        super().__init__(experiment_result)
        self.params_list = self._extract_parameters()
        self.df = self._create_dataframe()

    def _extract_parameters(self) -> List[Dict[str, Any]]:
        """Extract parameters from experiment results."""
        params_list = []
        for run in self.experiment_result.run_results:
            params_list.append(run.metadata.get("Prompt Parameters", {}))
        return params_list

    def _create_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from experiment results."""
        df = pd.DataFrame(self.params_list)
        df['score'] = [run.score for run in self.experiment_result.run_results]
        return df

    def create_interactive_visualization(self) -> None:
        """Create all interactive visualizations."""
        self.create_parameter_combination_heatmap()
        self.create_run_results_table()

    def create_parameter_combination_heatmap(self) -> None:
        """Create an interactive heatmap visualization for parameter combinations."""
        if not self.experiment_result or not self.experiment_result.run_results:
            print("No results to visualize.")
            return

        # Get parameter columns
        param_columns = [col for col in self.df.columns if col != 'score']
        if len(param_columns) < 2:
            print("Not enough parameters for heatmap visualization.")
            return

        # Create parameter selection widgets
        param1_dropdown = widgets.Dropdown(
            options=param_columns,
            description='Parameter 1:',
            value=param_columns[0]
        )

        param2_dropdown = widgets.Dropdown(
            options=param_columns,
            description='Parameter 2:',
            value=param_columns[1]
        )

        def update_heatmap(param1: str, param2: str) -> None:
            plt.figure(figsize=(12, 8))

            # Create pivot table for heatmap
            heatmap_data = self.df.pivot_table(
                values='score',
                index=param1,
                columns=param2,
                aggfunc='mean'
            )

            # Create heatmap
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt='.3f',
                cmap='viridis',
                cbar_kws={'label': 'Score'}
            )

            plt.title(f'Parameter Combination Heatmap\n{param1} vs {param2}')
            plt.xlabel(param2)
            plt.ylabel(param1)
            plt.tight_layout()
            plt.show()

        # Create interactive widget
        widgets.interact(
            update_heatmap,
            param1=param1_dropdown,
            param2=param2_dropdown
        )

    def create_run_results_table(self) -> None:
        """Create an interactive table visualization for run results."""
        if not self.experiment_result or not self.experiment_result.run_results:
            print("No results to visualize.")
            return

        # Extract run data
        run_data = []
        for i, run in enumerate(self.experiment_result.run_results):
            run_dict = {
                'Run #': i + 1,
                'Score': run.score,
                **run.metadata.get("Prompt Parameters", {}),
            }

            # Add metric scores if available
            eval_result = run.metadata.get("Custom Evaluator Results", [{}])[0]
            if "scores" in eval_result:
                for metric, score in eval_result["scores"].items():
                    run_dict[f'{metric} Score'] = score

            run_data.append(run_dict)

        # Create DataFrame for table
        table_df = pd.DataFrame(run_data)

        # Create column selection widget
        columns = list(table_df.columns)
        column_selector = widgets.SelectMultiple(
            options=columns,
            value=columns[:5],  # Default to first 5 columns
            description='Columns:',
            layout={'width': 'max-content'}
        )

        # Create sort column widget
        sort_selector = widgets.Dropdown(
            options=['Run #'] + [col for col in columns if col != 'Run #'],
            value='Score',
            description='Sort by:',
        )

        # Create ascending/descending widget
        ascending_selector = widgets.Checkbox(
            value=False,
            description='Ascending',
        )

        def update_table(columns_to_show: List[str], sort_by: str, ascending: bool) -> None:
            # Sort DataFrame
            sorted_df = table_df.sort_values(by=sort_by, ascending=ascending)

            # Select columns
            display_df = sorted_df[list(columns_to_show)]

            # Convert to HTML with styling
            styled_df = display_df.style\
                .set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#f0f0f0')]},
                    {'selector': 'td', 'props': [('padding', '8px')]},
                ])\
                .format(precision=3)

            display(HTML(styled_df.to_html()))

        # Create interactive widget
        widgets.interact(
            update_table,
            columns_to_show=column_selector,
            sort_by=sort_selector,
            ascending=ascending_selector
        )
