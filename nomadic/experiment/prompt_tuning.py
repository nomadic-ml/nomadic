from typing import List, Optional, Dict, Union, Any
from itertools import product
import time
import re
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ipywidgets import interact, widgets
import torch
from transformers import BertTokenizer, BertModel
from openai import OpenAI

# Import evaluation functions
from nomadic.experiment.helpers.base_evaluator import (
    accuracy_evaluator,
    get_bert_embedding,
    evaluate_response,
    calculate_mean_score
)

# Import base experiment class
from nomadic.experiment.experiment_types import BaseExperiment

# Import dspy
import dspy  # Ensure dspy is installed in your environment

# Set your default model name here
DEFAULT_OPENAI_MODEL = "gpt-4o"  # or another model as needed

class PromptTuner:
    def __init__(
        self,
        prompt_tuning_approaches=["None"],
        prompt_tuning_topics=["hallucination-detection"],
        prompt_tuning_complexities=["None"],
        prompt_tuning_focuses=["None"],
        enable_logging=True,
        evaluation_dataset=[],
        use_iterative_optimization=False,
        optimal_threshold=0.8,  # Default threshold for optimization
        k=5,  # Default number of examples for few-shot learning
        optimizer_type="BootstrapFewShotWithRandomSearch"  # Default optimizer
    ):
        self.prompt_tuning_approaches = prompt_tuning_approaches
        self.prompt_tuning_topics = prompt_tuning_topics
        self.prompt_tuning_complexities = prompt_tuning_complexities
        self.prompt_tuning_focuses = prompt_tuning_focuses
        self.enable_logging = enable_logging
        self.evaluation_dataset = evaluation_dataset
        self.use_iterative_optimization = use_iterative_optimization
        self.optimal_threshold = optimal_threshold
        self.k = k
        self.optimizer_type = optimizer_type

        # Initialize the optimizer if DSPy optimization is enabled
        if self.use_iterative_optimization == "dspy":
            if self.optimizer_type == "BootstrapFewShotWithRandomSearch":
                from dspy.teleprompt import BootstrapFewShotWithRandomSearch
                # Configure the BootstrapFewShotWithRandomSearch optimizer
                self.optimizer = BootstrapFewShotWithRandomSearch(
                    metric=self.evaluate_variant,
                    max_bootstrapped_demos=self.k,
                    max_labeled_demos=self.k
                )
            else:
                raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

    def generate_prompt_variants(self, client, user_prompt_request, max_retries: int = 3, retry_delay: int = 5):
        """
        Generate prompt variants using either static or iterative optimization.

        Args:
            client: The client object for API calls.
            user_prompt_request: List of prompt templates.
            max_retries: Maximum number of retries for API calls.
            retry_delay: Delay between retries.

        Returns:
            List of generated prompt variants.
        """
        if self.use_iterative_optimization == "dspy":
            return self.generate_prompt_variants_optimized(client, user_prompt_request, max_retries, retry_delay)
        elif self.use_iterative_optimization == "iterative":
            return self.generate_prompt_variants_iterative(client, user_prompt_request, max_retries, retry_delay)
        elif self.use_iterative_optimization == "rl_agent":
            return self.generate_prompt_variants_rl_agent(client, user_prompt_request, max_retries, retry_delay)
        else:
            return self.generate_prompt_variants_static(client, user_prompt_request, max_retries, retry_delay)

    def generate_prompt_variants_static(self, client, user_prompt_request, max_retries: int = 3, retry_delay: int = 5):
        """
        Generate prompt variants using static method.

        Args:
            client: The client object for API calls.
            user_prompt_request: List of prompt templates.
            max_retries: Maximum number of retries for API calls.
            retry_delay: Delay between retries.

        Returns:
            List of generated prompt variants.
        """
        prompt_variants = []
        for template in user_prompt_request:  # Assume prompt_templates is now a list
            # Generate variants for each template
            variant = self.generate_prompt_variant(client, template, max_retries, retry_delay)
            prompt_variants.extend(variant)
        return prompt_variants

    def generate_prompt_variants_optimized(self, client, user_prompt_request, max_retries: int = 3, retry_delay: int = 5):
        """
        Generate prompt variants using iterative optimization with FewShotOptimizer from dspy.

        This method employs the FewShotOptimizer to iteratively refine prompt variants based on
        a set of parameters and an objective function. It aims to find the optimal combination
        of prompt tuning approaches, complexities, and focuses for each given template.

        Args:
            client: The client object for API calls.
            user_prompt_request: List of prompt templates to be optimized.
            max_retries: Maximum number of retries for API calls to mitigate transient errors.
            retry_delay: Delay between retries in seconds to avoid overwhelming the API.
        """
        prompt_variants = []
        for template in user_prompt_request:
            # Compile the optimizer with the current template
            compiled_program = self.optimizer.compile(trainset=[(template, None)])

            # Generate optimized variant
            variant = compiled_program(template)
            prompt_variants.append(variant)

        return prompt_variants

    def generate_prompt_variants_iterative(self, client, user_prompt_request, max_retries: int = 3, retry_delay: int = 5):
        """
        Generate prompt variants using an iterative optimization method without dspy.

        This method uses a simple iterative approach to refine prompt variants based on
        a set of parameters and an objective function. It aims to find an improved combination
        of prompt tuning approaches, complexities, and focuses for each given template.

        Args:
            client: The client object for API calls.
            user_prompt_request: List of prompt templates to be optimized.
            max_retries: Maximum number of retries for API calls to mitigate transient errors.
            retry_delay: Delay between retries in seconds to avoid overwhelming the API.

        Returns:
            List of optimized prompt variants, one for each input template.
        """
        prompt_variants = []
        for template in user_prompt_request:
            # Ensure template is a string for consistency in processing
            if not isinstance(template, str):
                if self.enable_logging:
                    print(f"Warning: Template '{template}' is not a string. Converting to string.")
                template = str(template)

            # Define the parameter space for optimization
            param_space = {
                'approach': self.prompt_tuning_approaches,
                'complexity': self.prompt_tuning_complexities,
                'focus': self.prompt_tuning_focuses,
            }

            # Define the objective function for optimization
            def objective_function(**params):
                # Update the current parameters with those provided by the optimizer
                self.prompt_tuning_approaches = [params['approach']]
                self.prompt_tuning_complexities = [params['complexity']]
                self.prompt_tuning_focuses = [params['focus']]

                # Generate a prompt variant using the current parameters
                variants = self.generate_prompt_variant(client, template, max_retries, retry_delay)
                if not variants:
                    return 0.0  # Return zero score if no variant is generated
                variant = variants[0]

                # Evaluate the variant and return its score
                score = self.evaluate_variant(variant)
                return score

            # Initialize the selected optimizer
            if self.optimizer_type == "BootstrapFewShotWithRandomSearch":
                optimizer = self.optimizer
            else:
                raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

            # Perform optimization to find the best parameters
            if self.use_iterative_optimization == "dspy":
                optimized_program = self.optimizer.compile(
                    student=lambda x: objective_function(**x),
                    trainset=self.trainset,  # Use actual trainset
                )
            elif self.use_iterative_optimization == "iterative":
                optimized_program = self.iterative_optimization(objective_function)
            else:
                optimized_program = objective_function

            best_params = optimized_program.best_params
            best_score = optimized_program.best_score

            if self.enable_logging:
                print(f"Best parameters found: {best_params} with score: {best_score}")

            # Update the instance variables with the best found parameters
            self.prompt_tuning_approaches = [best_params['approach']]
            self.prompt_tuning_complexities = [best_params['complexity']]
            self.prompt_tuning_focuses = [best_params['focus']]

            # Generate the best prompt variant using the optimal parameters
            best_variants = self.generate_prompt_variant(client, template, max_retries, retry_delay)
            if best_variants:
                prompt_variants.extend(best_variants)
            else:
                prompt_variants.append(template)  # Fallback to the original template if generation fails

        return prompt_variants

    def generate_prompt_variants_rl_agent(self, client, user_prompt_request, max_retries: int = 3, retry_delay: int = 5):
        """
        Generate prompt variants using an iterative optimization method with an RL agent.

        This method uses a reinforcement learning agent to iteratively refine prompt variants based on
        a defined reward function, which is the evaluation score of the prompt. The agent continues to
        improve the prompt until an optimal score is achieved or the maximum iterations are reached.

        Args:
            client: The client object for API calls.
            user_prompt_request: List of prompt templates to be optimized.
            max_retries: Maximum number of retries for API calls.
            retry_delay: Delay between retries.

        Returns:
            List of optimized prompt variants, one for each input template.
        """
        prompt_variants = []
        max_iterations = 50  # Maximum number of iterations for the RL agent
        epsilon = 0.01  # Threshold for convergence

        for template in user_prompt_request:
            current_prompt = template
            best_prompt = current_prompt
            best_score = float('-inf')
            iteration = 0

            while iteration < max_iterations:
                if self.enable_logging:
                    print(f"\nIteration {iteration + 1}/{max_iterations}")
                    print(f"Current Prompt:\n{current_prompt}\n")

                # State: Current prompt
                state = current_prompt

                # Evaluate the current prompt to get the reward
                reward = self.evaluate_variant(state)
                if self.enable_logging:
                    print(f"Evaluation Score: {reward}")

                # Update the best prompt if the reward is better
                if reward > best_score:
                    best_score = reward
                    best_prompt = state

                # Check for convergence
                if reward >= 1.0 - epsilon:
                    if self.enable_logging:
                        print("Optimal prompt achieved.")
                    break

                # Action: Modify the prompt based on feedback
                feedback = self.get_feedback(state)
                action = self.modify_prompt_with_feedback(client, state, feedback, max_retries, retry_delay)

                # Update the prompt (next state)
                current_prompt = action
                iteration += 1

            prompt_variants.append(best_prompt)

        return prompt_variants

    def generate_prompt_variant(self, client, user_prompt_request, max_retries, retry_delay):
        # Create a list to store all generated prompt variants
        full_prompt_variants = []
        client = OpenAI()

        if (
            self.prompt_tuning_approaches == ["None"]
            and self.prompt_tuning_complexities == ["None"]
            and self.prompt_tuning_focuses == ["None"]
        ):
            return [user_prompt_request]

        # Generate all combinations of the specified areas
        combinations = product(
            self.prompt_tuning_approaches,
            self.prompt_tuning_topics,
            self.prompt_tuning_complexities,
            self.prompt_tuning_focuses,
        )

        # Iterate over all possible combinations
        for approach, topic, complexity, focus in combinations:
            if self.enable_logging:
                print(
                    f"\nGenerating prompt for: Approach={approach}, Topic={topic}, Complexity={complexity}, Focus={focus}"
                )

            # Create the system message based on the current combination
            system_message = self._create_system_message(
                user_prompt_request, approach, complexity, focus, topic
            )
            if self.enable_logging:
                print("system message")
                print(system_message)
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Generate the prompt variant
                    response = client.chat.completions.create(
                        model=DEFAULT_OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": system_message},
                            {
                                "role": "user",
                                "content": "Generate the prompt variant.",
                            },
                        ],
                        temperature=0.7,
                    )

                    generated_prompt = response.choices[0].message.content.strip()

                    # If the approach is 'few-shot', generate and incorporate examples
                    if approach == "few-shot":
                        examples = self._generate_examples(
                            client, user_prompt_request, topic, complexity, focus
                        )
                        generated_prompt = self._incorporate_examples(
                            generated_prompt, examples
                        )

                    # Use only the generated prompt, don't append the original user_prompt_request
                    full_prompt = generated_prompt
                    full_prompt_variants.append(full_prompt)

                    if self.enable_logging:
                        print(f"Generated full prompt:\n{full_prompt[:500]}...")

                    break  # Exit retry loop on success

                except Exception as e:
                    print(f"Error generating prompt variant: {e}")
                    print(f"Retrying... ({retry_count + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_count += 1

            if retry_count == max_retries:
                print(f"Failed to generate prompt variant after {max_retries} attempts.")

        return full_prompt_variants

    def optimize_prompt(self, client, prompt_variant, max_retries, retry_delay):
        """
        Optimize the prompt variant using client API calls directly.

        Args:
            client: The client object.
            prompt_variant: The prompt variant to optimize.
            max_retries: Maximum number of retries.
            retry_delay: Delay between retries.

        Returns:
            Optimized prompt variant.
        """
        for attempt in range(max_retries):
            try:
                # Use client API for optimization
                response = client.chat.completions.create(
                    model=DEFAULT_OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are an expert prompt engineer."},
                        {"role": "user", "content": f"Improve the following prompt for better hallucination detection:\n\n{prompt_variant}\n\nOptimized Prompt:"},
                    ],
                    temperature=0.7,
                )
                optimized_prompt = response.choices[0].message.content.strip()
                return optimized_prompt
            except Exception as e:
                if self.enable_logging:
                    print(f"Error during prompt optimization: {e}")
                    print(f"Retrying... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)

        # If all retries fail, return the original prompt variant
        if self.enable_logging:
            print(f"Failed to optimize prompt variant after {max_retries} attempts.")
        return prompt_variant

    def _create_system_message(
        self, user_prompt_request: str, approach: str, complexity: str, focus: str, topic: str = "hallucination-detection"
    ) -> str:

        # Mapping topic to specific instructions
        topic_instructions = {
            "hallucination-detection": "hallucination detection"
        }

        selected_topic = topic_instructions.get(topic, "generation of a high-quality prompt according to the instruction provided")
        # Mapping approach to specific instructions
        approach_instructions = {
            "zero-shot": f"Directly respond to the query with no prior examples, ensuring clarity and focus on {selected_topic}.",
            "few-shot": f"Incorporate a structured setup with examples that outline similar cases of {selected_topic}.",
            "chain-of-thought": f"Develop a step-by-step reasoning process within the prompt to elucidate how one might do {selected_topic}.",
        }

        # Mapping complexity to specific instructions
        complexity_instructions = {
            "simple": "Craft the prompt using straightforward, unambiguous language to ensure it is easy to follow.",
            "complex": f"Construct a detailed and nuanced prompt, incorporating technical jargon and multiple aspects of {topic}.",
        }

        # Mapping task to specific instructions
        focus_instructions = {
            "fact-extraction": f"Direct the model to extract and verify facts potentially prone to {selected_topic}.",
            "action-points": f"Delineate clear, actionable steps that the user can take to do {selected_topic}.",
            "summarization": f"Summarize the key points necessary for effective {selected_topic}.",
            "language-simplification": "Simplify the complexity of the language used in the prompt to make it more accessible.",
            "tone-changes": "Adapt the tone to be more formal or informal, depending on the context or target audience.",
            "british-english-usage": "Utilize British spelling and stylistic conventions throughout the prompt.",
            "coherence": "Ensure that the prompt is coherent."
        }
        selected_focus = focus_instructions.get(focus, "Ensure that the prompt is coherent.")
        # Define the initial part of the message that describes the AI's specialization
        base_message = f"""
You are an AI assistant specialized in generating prompts for a system where the goal is {selected_topic}.
As the assistant, your goal is to create a prompt that integrates the following parameters effectively:

- Approach: Use {approach} when generating the prompt.
- Complexity: {complexity}
- Focus: {selected_focus}


Based on these settings, adjust the base user prompt provided below:
"""

        # Construct the final system message
        system_message = base_message + f"Base prompt:\n\n{user_prompt_request}\n\n"
        system_message += "\n".join(
            [
                approach_instructions.get(
                    approach,
                    "Implement a generic approach that is adaptable to any scenario.",
                ),
                complexity_instructions.get(
                    complexity,
                    "Use a moderate level of complexity suitable for general audiences.",
                ),
                focus_instructions.get(
                    focus, "Ensure that the prompt is coherent."
                ),
                ("The goal is " +
                topic_instructions.get(
                    topic, "to generate a high-quality prompt according to the instruction provided."
                ) + "."),
                f"\nEnsure the final prompt integrates all elements to maintain a coherent and focused approach to {topic}.",
            ]
        )

        return system_message

    def _generate_examples(
        self,
        client,
        user_prompt_request: str,
        topic: Optional[str],
        complexity: Optional[str],
        focus: Optional[str],
        max_retries: int = 3,
        retry_delay: int = 5,
    ) -> List[Dict[str, str]]:
        system_message = f"""
Generate 3 concise input-output pairs for few-shot learning on {topic}.
Tailor examples to these parameters:
- Complexity: {complexity}
- Focus: {focus}

Base prompt: {user_prompt_request}

Format:
Input:
Query: [query text]
Context: [context text]
Response: [response text]

Output: [Faithful/Not faithful/Refusal]

Ensure examples reflect various {topic} scenarios.
"""

        retry_count = 0
        while retry_count < max_retries:
            try:
                response = client.chat.completions.create(
                    model=DEFAULT_OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": "Generate the examples."},
                    ],
                    temperature=0.7,
                )

                examples_text = response.choices[0].message.content.strip()
                examples = []

                example_pairs = re.findall(
                    r"Input:\nQuery: (.*?)\nContext: (.*?)\nResponse: (.*?)\n\nOutput: (.*?)(?=\n\nInput:|$)",
                    examples_text,
                    re.DOTALL,
                )
                for query, context, response_text, output in example_pairs:
                    examples.append(
                        {
                            "query": query.strip(),
                            "context": context.strip(),
                            "response": response_text.strip(),
                            "output": output.strip(),
                        }
                    )
                return examples

            except Exception as e:
                if self.enable_logging:
                    print(f"Error generating examples: {e}")
                    print(f"Retrying... ({retry_count + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_count += 1

        if self.enable_logging:
            print(f"Failed to generate examples after {max_retries} attempts.")
        return []

    def _incorporate_examples(
        self, generated_prompt: str, examples: List[Dict[str, str]]
    ) -> str:
        examples_text = ""
        for i, example in enumerate(examples, 1):
            example_text = f"""
Example {i}:
Input:
Query: {example['query']}
Context: {example['context']}
Response: {example['response']}

Output: {example['output']}
"""
            examples_text += example_text

        # Incorporate the examples into the generated prompt
        full_prompt = f"{generated_prompt}\n\n{examples_text}"
        return full_prompt

    def evaluate_variant(self, variant):
        # Use the base_evaluator's accuracy_evaluator function to evaluate the variant
        if self.evaluation_dataset:
            evaluation_sample = self.evaluation_dataset[0]
            generated_answer = variant
            ground_truth = evaluation_sample.get('answer')
            from src.helpers.evaluation.base_evaluator import accuracy_evaluator
            evaluation_result = accuracy_evaluator(
                generated_answer=generated_answer,
                ground_truth=ground_truth,
                evaluation_dataset=self.evaluation_dataset
            )
            return evaluation_result['similarity_score']
        else:
            # If no evaluation dataset, return a default score
            return 0.0

    def get_feedback(self, prompt):
        """
        Generate feedback for the given prompt.

        Args:
            prompt: The prompt to evaluate.

        Returns:
            Feedback string based on evaluation.
        """
        # Use accuracy_evaluator to evaluate the prompt
        if self.evaluation_dataset and len(self.evaluation_dataset) > 0:
            evaluation_sample = self.evaluation_dataset[0]
            ground_truth = evaluation_sample.get('answer', '')

            evaluation_result = accuracy_evaluator(
                generated_answer=prompt,
                ground_truth=ground_truth,
                evaluation_dataset=self.evaluation_dataset
            )

            # Construct feedback from evaluation metrics
            feedback = f"Similarity score: {evaluation_result['similarity_score']:.2f}"
            if 'evaluation_metrics' in evaluation_result:
                metrics = evaluation_result['evaluation_metrics']
                feedback += f"\nExact match: {metrics.get('exact_match', 0)}"
                if 'bert_similarity' in metrics:
                    feedback += f"\nBERT similarity: {metrics['bert_similarity']:.2f}"
            return feedback
        return "No evaluation dataset available for feedback"

    def modify_prompt_with_feedback(self, client, prompt, feedback, max_retries, retry_delay):
        """
        Modify the prompt based on the feedback.

        Args:
            client: The client object for API calls.
            prompt: The current prompt.
            feedback: Feedback string to guide the modification.
            max_retries: Maximum number of retries.
            retry_delay: Delay between retries.

        Returns:
            Modified prompt.
        """
        system_message = f"""
You are an expert prompt engineer. Improve the following prompt based on the feedback provided.

Original Prompt:
{prompt}

Feedback:
{feedback}

Provide the improved prompt, ensuring it addresses the issues mentioned.
"""

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=DEFAULT_OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": "Provide the improved prompt."},
                    ],
                    temperature=0.7,
                )
                new_prompt = response.choices[0].message.content.strip()
                return new_prompt
            except Exception as e:
                if self.enable_logging:
                    print(f"Error modifying prompt: {e}")
                    print(f"Retrying... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)

        if self.enable_logging:
            print(f"Failed to modify prompt after {max_retries} attempts.")
        return prompt  # Return the original prompt if unable to modify

    def update_params(self, params: Dict[str, Any]):
        if "prompt_tuning_approach" in params:
            self.prompt_tuning_approaches = [params["prompt_tuning_approach"]]
        if "prompt_tuning_complexity" in params:
            self.prompt_tuning_complexities = [params["prompt_tuning_complexity"]]
        if "prompt_tuning_focus" in params:
            self.prompt_tuning_focuses = [params["prompt_tuning_focus"]]
        if "prompt_tuning_topic" in params:
            self.prompt_tuning_topics = [params["prompt_tuning_topic"]]
        if "enable_logging" in params:
            self.enable_logging = params["enable_logging"]
        if "evaluation_dataset" in params:
            self.evaluation_dataset = params["evaluation_dataset"]

    def __str__(self):
        return f"PromptTuner(approaches={self.prompt_tuning_approaches}, topics={self.prompt_tuning_topics}, complexities={self.prompt_tuning_complexities}, focuses={self.prompt_tuning_focuses})"

    def __repr__(self):
        return self.__str__()





class RLSimulation(BaseExperiment):
    def __init__(self, param_fn=None, params=None, fixed_param_dict=None, enable_logging=True):
        super().__init__(
            model="gpt-4",  # Default model
            evaluation_dataset=[],  # Will be populated during simulation
            use_iterative_optimization=True,
            param_fn=param_fn,
            params=params,
            fixed_param_dict=fixed_param_dict,
            enable_logging=enable_logging
        )
        self.prompt_tuner = None  # Will be initialized in setup
        self.client = None
        self.initial_prompt = None
        self.evaluation_metrics = None
        self.openai_api_key = None
        self.prompts = []
        self.scores = []
        self.feedbacks = []
        self.iterations = 0
        self.max_iterations = 10  # You can adjust this
        self.epsilon = 0.01  # Convergence threshold

    def _get_responses(self, **kwargs) -> tuple:
        """Override base method to implement RL-specific response generation."""
        # Initialize prompt tuner if not already done
        if not self.prompt_tuner:
            self.prompt_tuner = PromptTuner(
                enable_logging=self.enable_logging,
                evaluation_dataset=self.evaluation_dataset
            )
            self.initial_prompt = kwargs.get('initial_prompt', '')
            self.evaluation_metrics = kwargs.get('evaluation_metrics', [])
            self.client = kwargs.get('client')
            self.openai_api_key = kwargs.get('openai_api_key')

        current_prompt = self.initial_prompt
        best_prompt = current_prompt
        best_score = float('-inf')
        responses = []
        eval_questions = []
        prompt_variants = []

        for iteration in range(self.max_iterations):
            if self.enable_logging:
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
                print(f"Current Prompt:\n{current_prompt}\n")

            # Evaluate the current prompt using base_evaluator's accuracy_evaluator
            if self.evaluation_dataset and len(self.evaluation_dataset) > 0:
                evaluation_sample = self.evaluation_dataset[0]
                ground_truth = evaluation_sample.get('answer', '')
                evaluation_result = accuracy_evaluator(
                    generated_answer=current_prompt,
                    ground_truth=ground_truth,
                    evaluation_dataset=self.evaluation_dataset
                )
                score = evaluation_result['similarity_score']
                feedback = f"Similarity score: {score:.2f}"
                if 'evaluation_metrics' in evaluation_result:
                    metrics = evaluation_result['evaluation_metrics']
                    feedback += f"\nExact match: {metrics.get('exact_match', 0)}"
                    if 'bert_similarity' in metrics:
                        feedback += f"\nBERT similarity: {metrics['bert_similarity']:.2f}"
            else:
                score = 0.5  # Default score if no evaluation dataset
                feedback = "No evaluation dataset available for feedback"

            # Store the data for visualization
            self.prompts.append(current_prompt)
            self.scores.append(score)
            self.feedbacks.append(feedback)
            self.iterations += 1

            responses.append(current_prompt)
            eval_questions.append(feedback)
            prompt_variants.append(current_prompt)

            # Update the best prompt if the score is better
            if score > best_score:
                best_score = score
                best_prompt = current_prompt

            # Check for convergence
            if score >= 1.0 - self.epsilon:
                if self.enable_logging:
                    print("Optimal prompt achieved.")
                break

            # Generate a new prompt variant using the feedback
            current_prompt = self.prompt_tuner.modify_prompt_with_feedback(
                self.client, current_prompt, feedback, max_retries=3, retry_delay=2
            )

        return responses, eval_questions, prompt_variants

    def _evaluate_responses(self, responses: List[str], eval_questions: List[str]) -> List[float]:
        """Override base method to implement RL-specific evaluation."""
        if not self.evaluation_metrics:
            return [0.5] * len(responses)  # Default neutral score if no metrics

        scores = []
        for response, question in zip(responses, eval_questions):
            # Get the ground truth from evaluation dataset if available
            ground_truth = None
            if self.evaluation_dataset:
                for entry in self.evaluation_dataset:
                    if entry['query'] == question:
                        ground_truth = entry['answer']
                        break

            if ground_truth:
                result = accuracy_evaluator(
                    generated_answer=response,
                    ground_truth=ground_truth,
                    evaluation_dataset=self.evaluation_dataset
                )
                scores.append(result['similarity_score'])
            else:
                scores.append(0.5)  # Default score if no ground truth found
        return scores

    def plot_scores(self):
        """Visualize the progression of scores during RL optimization."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.iterations + 1), self.scores, marker='o', linestyle='-')
        plt.title('Prompt Evaluation Score Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Evaluation Score')
        plt.xticks(range(1, self.iterations + 1))
        plt.ylim(0, 1)
        plt.grid(True)
        plt.show()

    def display_prompts(self):
        """Display interactive prompt history with scores and feedback."""
        def show_prompt(iteration):
            index = iteration - 1
            print(f"Iteration {iteration}")
            print(f"Prompt:\n{self.prompts[index]}\n")
            print(f"Evaluation Score: {self.scores[index]}")
            print(f"Feedback:\n{self.feedbacks[index]}\n")

        interact(
            show_prompt,
            iteration=widgets.IntSlider(
                min=1, max=self.iterations, step=1, value=1,
                description='Iteration',
                style={'description_width': 'initial'}
            )
        )

from base_evaluator import get_bert_embedding, accuracy_evaluator

def transform_eval_dataset_to_eval_json(eval_dataset):
    eval_json = {
        "queries": {},
        "responses": {}
    }
    # Loop through each entry in eval_dataset
    for idx, entry in enumerate(eval_dataset, start=1):
        query_key = f"query{idx}"
        eval_json["queries"][query_key] = entry['query']
        eval_json["responses"][query_key] = entry['answer']
    return eval_json
