import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import ipywidgets as widgets
from IPython.display import display, HTML

def plot_scores(scores: List[float], title: str = "Score Progress"):
    """
    Plot the progression of scores over iterations.

    Args:
        scores: List of scores from each iteration
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(scores, marker='o')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()

def display_prompts(prompts: List[str], scores: Optional[List[float]] = None):
    """
    Create an interactive display of prompts with their corresponding scores.

    Args:
        prompts: List of prompts from each iteration
        scores: Optional list of corresponding scores
    """
    if not prompts:
        print("No prompts to display.")
        return

    # Create iteration slider
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(prompts) - 1,
        step=1,
        description='Iteration:',
        continuous_update=False
    )

    def display_prompt(iteration):
        prompt = prompts[iteration]
        score_text = f"\nScore: {scores[iteration]:.3f}" if scores and iteration < len(scores) else ""

        html_content = f"""
        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
            <h3>Iteration {iteration}</h3>
            <pre style="white-space: pre-wrap;">{prompt}</pre>
            <p><strong>{score_text}</strong></p>
        </div>
        """
        display(HTML(html_content))

    # Create interactive widget
    widgets.interact(display_prompt, iteration=slider)
