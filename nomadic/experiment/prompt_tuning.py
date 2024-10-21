from typing import List, Optional, Dict, Union, Any
from itertools import product
import time
import re
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# Import the OpenAI class as per your code
from openai import OpenAI  # Ensure this import aligns with your environment

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
        # Use the custom_evaluate function to evaluate the variant
        if self.evaluation_dataset:
            evaluation_sample = self.evaluation_dataset[0]
            generated_answer = variant
            ground_truth = evaluation_sample.get('answer')
            evaluation_metrics = [
                {"metric": "Accuracy", "weight": 1.0}
            ]
            openai_api_key = None  # Replace with your OpenAI API key if necessary
            evaluation_result = custom_evaluate(
                response=generated_answer,
                evaluation_metrics=evaluation_metrics,
                openai_api_key=openai_api_key,
            )
            return evaluation_result['overall_score']
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
        # For simplicity, we can use the evaluation explanation as feedback
        evaluation_metrics = [
            {"metric": "Accuracy", "weight": 1.0}
        ]
        openai_api_key = None  # Replace with your OpenAI API key if necessary
        evaluation_result = custom_evaluate(
            response=prompt,
            evaluation_metrics=evaluation_metrics,
            openai_api_key=openai_api_key,
        )
        feedback = evaluation_result.get('explanation', '')
        return feedback

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

def custom_evaluate(
    response: str,
    evaluation_metrics: List[Dict[str, Union[str, float]]],
    openai_api_key: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: int = 5,  # seconds
) -> dict:
    if openai_api_key:
        # Set the API key if provided
        import openai
        openai.api_key = openai_api_key

    metrics = [metric["metric"] for metric in evaluation_metrics]
    weights = {metric["metric"]: metric["weight"] for metric in evaluation_metrics}

    metrics_prompt = "\n".join([f"{i+1}. {metric}" for i, metric in enumerate(metrics)])
    metrics_format = "\n".join([f'"{metric}": [SCORE]' for metric in metrics])

    evaluation_prompt = f"""
You are a highly critical expert in evaluating responses to prompts. Your task is to rigorously assess the quality and accuracy of a generated response. Your evaluation must be extremely thorough, critical, and unbiased.

Evaluate the following response based on these criteria, scoring each from 0 to 100:

{metrics_prompt}

Crucial evaluation guidelines:
1. Accuracy is paramount. Any factual errors or misleading information should result in a very low accuracy score.
2. Be exceptionally critical. High scores (above 80) should be rare and only given for truly outstanding responses.
3. Consider the real-world implications of the advice. Would it be genuinely helpful and appropriate for a context?
4. Look for nuance and depth. Surface-level or overly generic advice should not score highly.
5. Evaluate the coherence and logical flow of the response.
6. Check for potential biases or unfounded assumptions in the advice.
7. Assess whether the response addresses all aspects of the original query or instruction.

Response to evaluate:
{response}

Provide your evaluation in the following JSON format:
{{
    "scores": {{
{metrics_format}
    }},
    "overall_score": [OVERALL_SCORE],
    "critical_analysis": "[Your detailed critical analysis, highlighting strengths, weaknesses, and specific issues]"
}}

CRITICAL INSTRUCTIONS:
1. Scores MUST be integers between 0 and 100. Use the full range appropriately.
2. The overall_score should reflect a weighted consideration of all criteria, not just an average.
3. In your critical_analysis, provide specific examples from the response to justify your scores.
4. Be explicit about any shortcomings, inaccuracies, or areas for improvement.
5. Consider potential negative consequences of following the advice, if any.
6. If the response is clearly inappropriate or harmful, do not hesitate to give very low scores.

Remember: Your evaluation could influence real decisions. Be as critical and thorough as possible.
"""

    retry_count = 0
    while retry_count < max_retries:
        try:
            import openai
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI assistant tasked with critically evaluating responses. Your evaluation must be thorough, unbiased, and reflect high standards of accuracy and quality.",
                },
                {"role": "user", "content": evaluation_prompt},
            ]
            completion = openai.ChatCompletion.create(
                model=DEFAULT_OPENAI_MODEL,
                messages=messages,
                temperature=0,
                max_tokens=1000,
            )
            evaluation_result = completion.choices[0].message.content.strip()

            # Parse the evaluation result
            evaluation_result_cleaned = evaluation_result.replace("\n", "")
            result = json.loads(evaluation_result_cleaned)
            scores = result.get("scores", {})
            overall_score = result.get("overall_score", 0)
            analysis = result.get("critical_analysis", "")

            # Ensure all scores are integers and within range
            for metric in metrics:
                if metric not in scores or not isinstance(scores[metric], (int, float)):
                    print(
                        f"Warning: Invalid or missing score for metric '{metric}'. Assigning default score of 0."
                    )
                    scores[metric] = 0
                else:
                    scores[metric] = round(scores[metric])
                    scores[metric] = max(0, min(scores[metric], 100))

            # Scale scores to be between 0 and 1
            scaled_scores = {metric: score / 100 for metric, score in scores.items()}

            # Ensure overall score is an integer and within range, then scale
            if not isinstance(overall_score, (int, float)):
                print(
                    f"Warning: Invalid overall score: {overall_score}. Calculating from individual scores."
                )
                overall_score = sum(scores.values()) / len(scores)
            else:
                overall_score = round(overall_score)
                overall_score = max(0, min(overall_score, 100))
            scaled_overall_score = overall_score / 100

            return {
                "scores": scaled_scores,
                "overall_score": scaled_overall_score,
                "explanation": analysis,
                "evaluation_prompt": evaluation_prompt,
            }

        except Exception as e:
            print(f"Error during evaluation: {e}")
            print(f"Retrying... ({retry_count + 1}/{max_retries})")
            time.sleep(retry_delay)
            retry_count += 1

    print(f"Failed to evaluate after {max_retries} attempts.")
    return {
        "scores": {metric: 0 for metric in metrics},
        "overall_score": 0,
        "explanation": "Evaluation failed.",
        "evaluation_prompt": evaluation_prompt,
    }

def custom_evaluate_hallucination(
    generated_answer: str,
    ground_truth: Optional[str],
    evaluation_dataset: List[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
    """
    Evaluate the generated answer by comparing it with the ground truth and
    dynamically match the closest answer from the given evaluation dataset using TF-IDF based similarity.

    Args:
    - generated_answer: The generated answer to be evaluated.
    - ground_truth: The correct ground truth answer.
    - evaluation_dataset: A list of dictionaries, each containing 'query', 'context', 'response', and 'answer'.

    Returns:
    - A dictionary with the normalized answer, analysis, correctness, and score.
    """
    if not evaluation_dataset:
        return {
            "answer": "Not provided",
            "analysis": "Evaluation dataset is missing or not provided.",
            "is_correct": False,
            "score": 0.0,
        }

    if not isinstance(ground_truth, str):
        return {
            "answer": "Not provided",
            "analysis": "The ground truth is not provided or is not a valid string.",
            "is_correct": False,
            "score": 0.0,
        }

    # Prepare the data for TF-IDF vectorization
    evaluation_labels = [entry["answer"].strip().lower() for entry in evaluation_dataset]
    all_answers = [generated_answer.strip().lower()] + evaluation_labels

    # Use TF-IDF to create embeddings
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_answers)

    # Calculate cosine similarity between generated answer and all evaluation labels
    generated_vec = tfidf_matrix[0].toarray()
    label_vecs = tfidf_matrix[1:].toarray()

    cosine_similarities = np.dot(label_vecs, generated_vec.T).flatten() / (
        np.linalg.norm(label_vecs, axis=1) * np.linalg.norm(generated_vec)
    )

    best_match_index = np.argmax(cosine_similarities)
    highest_similarity = cosine_similarities[best_match_index]
    best_match = evaluation_labels[best_match_index]

    normalized_answer = best_match if highest_similarity > 0.75 else "Not Faithful"
    is_correct = normalized_answer.lower() == ground_truth.lower()
    score = 1.0 if is_correct else 0.0

    analysis = (
        f"The generated answer '{generated_answer}' is most similar to '{best_match}' "
        f"with a similarity score of {highest_similarity:.2f}. "
    )
    analysis += (
        "The answer is considered correct as it matches the ground truth."
        if is_correct
        else f"The answer is considered incorrect as it does not match the ground truth '{ground_truth}'."
    )

    return {
        "answer": normalized_answer,
        "analysis": analysis,
        "is_correct": is_correct,
        "score": score,
    }
