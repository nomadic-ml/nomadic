from typing import List, Optional, Dict, Union
from openai import OpenAI
from pydantic import BaseModel, Field

from nomadic.model import DEFAULT_OPENAI_MODEL


from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
import re
import time

from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI


class PromptTuner(BaseModel):
    prompting_approaches: List[str] = Field(
        default=["None"],  # Default to ['None'] to indicate no tuning
        description="List of prompting approaches to use. If ['None'], no tuning is applied.",
    )
    prompt_complexities: List[str] = Field(
        default=["None"],  # Default to ['None'] to indicate no tuning
        description="List of prompt complexities to use. If ['None'], no tuning is applied.",
    )
    prompt_focuses: List[str] = Field(
        default=["None"],  # Default to ['None'] to indicate no tuning
        description="List of prompt focuses to use. If ['None'], no tuning is applied.",
    )
    enable_logging: bool = Field(
        default=True,
        description="Flag to enable or disable print logging.",
    )

    def generate_prompt_variants(
        self, user_prompt_request: str, api_key: Optional[str] = None
    ) -> List[str]:
        if not api_key:
            raise ValueError("OpenAI API key is required for prompt tuning.")

        # Check for ['None'] values and return the normal prompt if any parameter is ['None']
        if (
            self.prompting_approaches == ["None"]
            or self.prompt_complexities == ["None"]
            or self.prompt_focuses == ["None"]
        ):
            if self.enable_logging:
                print(
                    "One or more prompt tuning parameters are set to ['None']. Returning the normal prompt."
                )
            return [user_prompt_request]  # Return normal prompt without tuning

        client = OpenAI(api_key=api_key)
        prompt_variants = []

        # Original code for generating prompt variants
        for approach in self.prompting_approaches:
            for complexity in self.prompt_complexities:
                for focus in self.prompt_focuses:
                    if self.enable_logging:
                        print(
                            f"\nGenerating prompt for: Approach={approach}, Complexity={complexity}, Focus={focus}"
                        )

                    system_message = self._create_system_message(
                        user_prompt_request, approach, complexity, focus
                    )

                    try:
                        response = client.chat.completions.create(
                            model="gpt-4",
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

                        if approach == "few-shot":
                            examples = self._generate_examples(
                                client, user_prompt_request, complexity, focus
                            )
                            generated_prompt = self._incorporate_examples(
                                generated_prompt, examples
                            )

                        prompt_variants.append(generated_prompt)

                        if self.enable_logging:
                            print(f"Generated prompt:\n{generated_prompt[:500]}...")

                    except Exception as e:
                        if self.enable_logging:
                            print(f"Error generating prompt variant: {str(e)}")

        return prompt_variants

    def _create_system_message(
        self, user_prompt_request: str, approach: str, complexity: str, focus: str
    ) -> str:
        # If any of the parameters are 'None', return a default system message
        if approach == "None" or complexity == "None" or focus == "None":
            return "Standard prompt without tuning."

        base_message = f"""
        You are an AI assistant specialized in generating concise prompts.
        Generate a prompt based on these parameters:
        - Approach: {approach}
        - Complexity: {complexity}
        - Focus: {focus}

        User prompt:

        {user_prompt_request}

        Adjust the prompt accordingly:
        """

        approach_instructions = {
            "zero-shot": "Create a clear, concise prompt without examples.",
            "few-shot": "Include placeholders for concise examples like [EXAMPLE_1].",
            "chain-of-thought": "Encourage brief step-by-step reasoning.",
        }

        complexity_instructions = {
            "simple": "Use straightforward, concise language.",
            "complex": "Use detailed but concise instructions.",
        }

        focus_instructions = {
            "fact extraction": "Emphasize concise key facts.",
            "action points": "Focus on concise actionable insights.",
        }

        return base_message + "\n".join(
            [
                approach_instructions[approach],
                complexity_instructions[complexity],
                focus_instructions[focus],
                "\nEnsure the prompt reflects the approach, complexity, and focus.",
            ]
        )

    def _generate_examples(
        self,
        client,
        user_prompt_request: str,
        complexity: Optional[str],
        focus: Optional[str],
    ) -> List[str]:
        system_message = f"""
        Generate 3 concise input-output pairs for few-shot learning.
        Tailor examples to these parameters:
        - Complexity: {complexity}
        - Focus: {focus}

        Prompt: {user_prompt_request}

        Format:
        Input: [input text]
        Output: [output text]

        Ensure answers match input length.
        """
        if complexity is None or focus is None:
            return []

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": "Generate the examples."},
            ],
            temperature=0.7,
        )

        examples_text = response.choices[0].message.content.strip()
        examples = []

        # Parse the examples
        example_pairs = re.findall(
            r"Input: (.*?)Output: (.*?)(?=Input:|$)", examples_text, re.DOTALL
        )
        for input_text, output_text in example_pairs:
            examples.append(
                {"input": input_text.strip(), "output": output_text.strip()}
            )

        return examples

    def _incorporate_examples(self, generated_prompt: str, examples: List[str]) -> str:
        # Handle empty examples list
        if not examples:
            return generated_prompt
        for i, example in enumerate(examples, 1):
            example_text = f"Example {i}:\nInput: {example['input']}\nOutput: {example['output']}\n\n"
            prompt = generated_prompt.replace(f"[EXAMPLE_{i}]", example_text)
        return prompt

    def update_params(self, params: Dict[str, Any]):
        if "prompt_tuning_approach" in params:
            self.prompting_approaches = [params["prompt_tuning_approach"]]
        if "prompt_tuning_complexity" in params:
            self.prompt_complexities = [params["prompt_tuning_complexity"]]
        if "prompt_tuning_focus" in params:
            self.prompt_focuses = [params["prompt_tuning_focus"]]

    def __str__(self):
        return f"PromptTuner(approaches={self.prompting_approaches}, complexities={self.prompt_complexities}, focuses={self.prompt_focuses})"

    def __repr__(self):
        return self.__str__()


def custom_evaluate(
    response: str,
    evaluation_metrics: List[Dict[str, Union[str, float]]],
    openai_api_key: str,
    max_retries: int = 3,
    retry_delay: int = 5,  # seconds
) -> dict:
    metrics = [metric["metric"] for metric in evaluation_metrics]
    weights = {metric["metric"]: metric["weight"] for metric in evaluation_metrics}

    metrics_prompt = "\n".join([f"{i+1}. {metric}" for i, metric in enumerate(metrics)])
    metrics_format = "\n".join([f'"{metric}": [SCORE]' for metric in metrics])

    evaluation_prompt = f"""
    You are a highly critical expert in evaluating responses to prompts . Your task is to rigorously assess the quality and accuracy of a generated response. Your evaluation must be extremely thorough, critical, and unbiased.

    Evaluate the following response based on these criteria, scoring each from 0 to 100:

    {metrics_prompt}

    Crucial evaluation guidelines:
    1. Accuracy is paramount. Any factual errors or misleading information should result in a very low accuracy score.
    2. Be exceptionally critical. High scores (above 80) should be rare and only given for truly outstanding responses.
    3. Consider the real-world implications of the advice. Would it be genuinely helpful and appropriate for a financial context?
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

    Remember: Your evaluation could influence real financial decisions. Be as critical and thorough as possible.
    """

    client = OpenAI(api_key=openai_api_key)

    def get_evaluation_result(prompt):
        completion = client.chat.completions.create(
            model=DEFAULT_OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant tasked with critically evaluating responses. Your evaluation must be thorough, unbiased, and reflect high standards of accuracy and quality. Do not wrap the json codes in JSON markers",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content

    def parse_evaluation_result(evaluation_result: str):
        import json

        evaluation_result_cleaned = evaluation_result.replace("\n", "")
        try:
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

            return scaled_scores, scaled_overall_score, analysis

        except json.JSONDecodeError as e:
            print(f"Error parsing evaluation result: {e}")
            print(f"Raw evaluation result: {evaluation_result_cleaned}")
            return {metric: 0 for metric in metrics}, 0, "Error in evaluation."

    retry_count = 0
    while retry_count < max_retries:
        evaluation_result = get_evaluation_result(evaluation_prompt)
        scores, overall_score, analysis = parse_evaluation_result(evaluation_result)

        if all(metric in scores for metric in metrics) and overall_score is not None:
            break

        print(f"Retrying evaluation... ({retry_count + 1}/{max_retries})")
        time.sleep(retry_delay)
        retry_count += 1

    # Sanity check for obviously wrong answers
    if (
        "Accuracy" in scores
        and scores["Accuracy"] > 0.8
        and ("error" in response.lower() or "incorrect" in analysis.lower())
    ):
        print(
            "Warning: High accuracy score for potentially erroneous response. Adjusting scores."
        )
        scores["Accuracy"] = min(scores["Accuracy"], 0.2)
        analysis += (
            " Note: Accuracy score adjusted due to potential errors in the response."
        )

    # Calculate weighted score
    total_weight = sum(weights.values())
    weighted_score = (
        sum(scores.get(metric, 0) * weights.get(metric, 1) for metric in metrics)
        / total_weight
    )

    return {
        "scores": scores,
        "overall_score": weighted_score,
        "explanation": analysis,
        "evaluation_prompt": evaluation_prompt,
    }
