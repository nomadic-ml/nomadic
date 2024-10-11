from typing import List, Optional, Dict, Union, Any
from openai import OpenAI
from pydantic import BaseModel, Field
import re
import time
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import numpy as np
from itertools import product
from openai import OpenAI



# Mock implementation of DEFAULT_OPENAI_MODEL
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"



class PromptTuner(BaseModel):
    prompt_tuning_approaches: List[str] = Field(
        default=["None"],
        description="List of prompting approaches to use. If ['None'], no tuning is applied.",
    )
    prompt_tuning_topics: List[str] = Field(
        default=["hallucination-detection"],
        description="List of prompt topics which are the goals of the experiment. If ['None'], no tuning is applied.",
    )
    prompt_tuning_complexities: List[str] = Field(
        default=["None"],
        description="List of prompt complexities to use. If ['None'], no tuning is applied.",
    )
    prompt_tuning_focuses: List[str] = Field(
        default=["None"],
        description="List of prompt tasks to focus on. If ['None'], no tuning is applied.",
    )
    enable_logging: bool = Field(
        default=True,
        description="Flag to enable or disable print logging.",
    )
    evaluation_dataset: List[Dict[str, str]] = Field(
        default=[],
        description="List of evaluation examples.",
    )

    def generate_prompt_variants(self, client, user_prompt_request, max_retries: int = 3, retry_delay: int = 5):
        prompt_variants = []
        for template in user_prompt_request:  # Assume prompt_templates is now a list
            # Generate variants for each template
            variant = self.generate_prompt_variant(client, template, max_retries, retry_delay)
            prompt_variants.extend(variant)
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
            print("system message")
            print(system_message)
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Generate the prompt variant
                    response = client.chat.completions.create(
                        model="gpt-4o",
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
    def _create_system_message(
        self, user_prompt_request: str, approach: str, complexity: str, focus: str, topic: str = "hallucination-detection"
    ) -> str:

        # Mapping topic to specific instructions
        topic_instructions= {
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
            "action-points": f"Delinate clear, actionable steps that the user can take to do {selected_topic}.",
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

    Remember: Your evaluation could influence real  decisions. Be as critical and thorough as possible.
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
    score = 1.0 if is_correct  else 0.0

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

import torch
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Optional, Any
import numpy as np

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text: str) -> torch.Tensor:
    """Converts text into a BERT embedding vector."""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the mean of the token embeddings to create a single vector representation
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def accuracy_evaluator(
    generated_answer: str,
    ground_truth: Optional[str],
    evaluation_dataset: List[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate the generated answer by comparing it with the ground truth using BERT-based similarity.

    Args:
    - generated_answer: The generated answer to be evaluated.
    - ground_truth: The correct ground truth answer.
    - evaluation_dataset: (Optional) A list of dictionaries containing 'query', 'context', 'response', and 'answer'.

    Returns:
    - A dictionary with the normalized answer, analysis, and similarity score as a measure of accuracy.
    """
    if not isinstance(ground_truth, str):
        return {
            "answer": "Not provided",
            "analysis": "The ground truth is not provided or is not a valid string.",
            "score": 0.0,
        }

    # Get BERT embeddings for both the generated answer and the ground truth
    generated_vec = get_bert_embedding(generated_answer.lower())
    ground_truth_vec = get_bert_embedding(ground_truth.lower())

    # Calculate cosine similarity between the BERT embeddings
    similarity_score = torch.nn.functional.cosine_similarity(generated_vec, ground_truth_vec, dim=0).item()

    analysis = (
        f"The generated answer '{generated_answer}' has a similarity score of {similarity_score:.2f} "
        f"compared to the ground truth answer '{ground_truth}'. "
    )

    return {
        "answer": generated_answer,
        "analysis": analysis,
        "score": similarity_score,
    }
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
