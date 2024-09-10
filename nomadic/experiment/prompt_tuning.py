from typing import List, Optional, Dict, Union, Any
from openai import OpenAI
from pydantic import BaseModel, Field
import re
import time
import json
from sentence_transformers import SentenceTransformer, util
import random

# Mock implementation of DEFAULT_OPENAI_MODEL
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"

# Load a pre-trained model for computing sentence embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")


class PromptTuner(BaseModel):
    prompting_approaches: List[str] = Field(
        default=["None"],
        description="List of prompting approaches to use. If ['None'], no tuning is applied.",
    )
    prompt_complexities: List[str] = Field(
        default=["None"],
        description="List of prompt complexities to use. If ['None'], no tuning is applied.",
    )
    prompt_focuses: List[str] = Field(
        default=["None"],
        description="List of prompt focuses to use. If ['None'], no tuning is applied.",
    )
    enable_logging: bool = Field(
        default=True,
        description="Flag to enable or disable print logging.",
    )
    evaluation_dataset: List[Dict[str, str]] = Field(
        default=[],
        description="List of evaluation examples for hallucination detection.",
    )

    def generate_prompt_variants(
        self, user_prompt_request: str, api_key: Optional[str] = None
    ) -> List[str]:
        if not api_key:
            raise ValueError("OpenAI API key is required for prompt tuning.")

        if (
            self.prompting_approaches == ["None"]
            and self.prompt_complexities == ["None"]
            and self.prompt_focuses == ["None"]
        ):
            if self.enable_logging:
                print(
                    "All prompt tuning parameters are set to ['None']. Returning the normal prompt."
                )
            return [user_prompt_request]

        client = OpenAI(api_key=api_key)
        full_prompt_variants = []

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
                    print("system message")
                    print(system_message)
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
                        print("generated prompt")
                        print(generated_prompt)

                        if approach == "few-shot":
                            examples = self._generate_examples(
                                client, user_prompt_request, complexity, focus
                            )
                            generated_prompt = self._incorporate_examples(
                                generated_prompt, examples
                            )

                        # Apply the generated prompt variant to the full prompt
                        full_prompt = f"{generated_prompt}\n\n{user_prompt_request}"
                        full_prompt_variants.append(full_prompt)

                        if self.enable_logging:
                            print(f"Generated full prompt:\n{full_prompt[:500]}...")

                    except Exception as e:
                        if self.enable_logging:
                            print(f"Error generating prompt variant: {str(e)}")

        return full_prompt_variants

    def _create_system_message(
        self, user_prompt_request: str, approach: str, complexity: str, focus: str
    ) -> str:
        # Define the initial part of the message that describes the AI's specialization
        base_message = f"""
        You are an AI assistant specialized in generating prompts specifically for hallucination detection.
        The goal is to create a prompt that integrates the following parameters effectively:

        - Approach: {approach}
        - Complexity: {complexity}
        - Focus: {focus}

        Based on these settings, adjust the base user prompt provided below:
        """

        # Mapping approach to specific instructions
        approach_instructions = {
            "zero-shot": "Directly respond to the query with no prior examples, ensuring clarity and focus on hallucination detection.",
            "few-shot": "Incorporate a structured setup with examples that outline similar cases of hallucination detection.",
            "chain-of-thought": "Develop a step-by-step reasoning process within the prompt to elucidate how one might identify hallucinations.",
        }

        # Mapping complexity to specific instructions
        complexity_instructions = {
            "simple": "Craft the prompt using straightforward, unambiguous language to ensure it is easy to follow.",
            "complex": "Construct a detailed and nuanced prompt, incorporating technical jargon and multiple aspects of hallucination detection.",
        }

        # Mapping focus to specific instructions
        focus_instructions = {
            "fact extraction": "Ensure the prompt directs the model to extract and verify facts potentially prone to hallucinations.",
            "action points": "Focus the prompt on delineating clear, actionable steps that the user can take to detect or prevent hallucinations.",
            "summary": "Summarize the key points necessary for effective hallucination detection.",
            "simplify language": "Simplify the complexity of the language used in the prompt to make it more accessible.",
            "change tone": "Adapt the tone to be more formal or informal, depending on the context or target audience.",
            "British English usage": "Utilize British spelling and stylistic conventions throughout the prompt.",
        }

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
                    focus, "Focus broadly on hallucination detection strategies."
                ),
                "\nEnsure the final prompt integrates all elements to maintain a coherent and focused approach to hallucination detection.",
            ]
        )

        return system_message

    def _generate_examples(
        self,
        client,
        user_prompt_request: str,
        complexity: Optional[str],
        focus: Optional[str],
    ) -> List[Dict[str, str]]:
        system_message = f"""
        Generate 3 concise input-output pairs for few-shot learning in hallucination detection.
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

        Ensure examples reflect various hallucination scenarios.
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

        example_pairs = re.findall(
            r"Input:\nQuery: (.*?)\nContext: (.*?)\nResponse: (.*?)\n\nOutput: (.*?)(?=\n\nInput:|$)",
            examples_text,
            re.DOTALL,
        )
        for query, context, response, output in example_pairs:
            examples.append(
                {
                    "query": query.strip(),
                    "context": context.strip(),
                    "response": response.strip(),
                    "output": output.strip(),
                }
            )

        return examples

    def _incorporate_examples(
        self, generated_prompt: str, examples: List[Dict[str, str]]
    ) -> str:
        for i, example in enumerate(examples, 1):
            example_text = f"""
            Example {i}:
            Query: {example['query']}
            Context: {example['context']}
            Response: {example['response']}
            Judgment: {example['output']}

            """
            generated_prompt = generated_prompt.replace(f"[EXAMPLE_{i}]", example_text)
        return generated_prompt

    def update_params(self, params: Dict[str, Any]):
        if "prompt_tuning_approach" in params:
            self.prompting_approaches = [params["prompt_tuning_approach"]]
        if "prompt_tuning_complexity" in params:
            self.prompt_complexities = [params["prompt_tuning_complexity"]]
        if "prompt_tuning_focus" in params:
            self.prompt_focuses = [params["prompt_tuning_focus"]]
        if "enable_logging" in params:
            self.enable_logging = params["enable_logging"]
        if "evaluation_dataset" in params:
            self.evaluation_dataset = params["evaluation_dataset"]

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
    You are a highly critical expert in evaluating responses to prompts. Your task is to rigorously assess the quality and accuracy of a generated response. Your evaluation must be extremely thorough, critical, and unbiased.

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


def custom_evaluate_hallucination(
    generated_answer: str,
    ground_truth: Optional[str],
    evaluation_dataset: List[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate the generated answer by comparing it with the ground truth and
    dynamically match the closest answer from the given evaluation dataset.

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

    evaluation_labels = list(
        set([entry["answer"].strip().lower() for entry in evaluation_dataset])
    )
    generated_answer_normalized = generated_answer.strip().lower()

    generated_embedding = model.encode(
        generated_answer_normalized, convert_to_tensor=True
    )
    label_embeddings = model.encode(
        [label.lower() for label in evaluation_labels], convert_to_tensor=True
    )

    similarities = util.pytorch_cos_sim(generated_embedding, label_embeddings)[0]
    best_match_index = similarities.argmax().item()
    highest_similarity = similarities[best_match_index].item()
    best_match = evaluation_labels[best_match_index]

    normalized_answer = best_match if highest_similarity > 0.75 else "Not Faithful"
    is_correct = normalized_answer.lower() == ground_truth.lower()
    score = 1.0 if is_correct and random.random() > 0.05 else 0.0

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