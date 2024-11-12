from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel
import numpy as np
from statistics import mean, median
from transformers import BertTokenizer, BertModel
import torch
import json
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Below is Nomadic's Semantic Similarity based Evaluation.
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

def custom_evaluate(
    response: str,
    evaluation_metrics: List[Dict[str, Union[str, float]]],
    openai_api_key: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: int = 5,  # seconds
    model: str = "gpt-4"
) -> dict:
    """
    Evaluate a response using custom metrics with OpenAI's API.

    Args:
        response: The response to evaluate
        evaluation_metrics: List of metrics and their weights
        openai_api_key: Optional OpenAI API key
        max_retries: Maximum number of retries for API calls
        retry_delay: Delay between retries in seconds
        model: The OpenAI model to use for evaluation

    Returns:
        Dictionary containing evaluation scores and analysis
    """
    if openai_api_key:
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

Remember: Your evaluation could influence real decisions. Be as critical and thorough as possible.
"""

    retry_count = 0
    while retry_count < max_retries:
        try:
            import openai
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI assistant tasked with critically evaluating responses.",
                },
                {"role": "user", "content": evaluation_prompt},
            ]
            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=1000,
            )
            evaluation_result = completion.choices[0].message.content.strip()

            # Parse and validate the evaluation result
            evaluation_result_cleaned = evaluation_result.replace("\n", "")
            result = json.loads(evaluation_result_cleaned)
            scores = result.get("scores", {})
            overall_score = result.get("overall_score", 0)
            analysis = result.get("critical_analysis", "")

            # Process and validate scores
            processed_scores = {}
            for metric in metrics:
                if metric not in scores or not isinstance(scores[metric], (int, float)):
                    logging.warning(f"Invalid or missing score for metric '{metric}'. Using default score of 0.")
                    processed_scores[metric] = 0
                else:
                    processed_scores[metric] = max(0, min(round(float(scores[metric])), 100))

            # Scale scores to [0, 1] range
            scaled_scores = {metric: score / 100 for metric, score in processed_scores.items()}
            scaled_overall_score = max(0, min(round(float(overall_score)), 100)) / 100

            return {
                "scores": scaled_scores,
                "overall_score": scaled_overall_score,
                "explanation": analysis,
                "evaluation_prompt": evaluation_prompt,
            }

        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            if retry_count < max_retries - 1:
                logging.info(f"Retrying... ({retry_count + 1}/{max_retries})")
                time.sleep(retry_delay)
            retry_count += 1

    logging.error(f"Failed to evaluate after {max_retries} attempts.")
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
    Evaluate the generated answer for hallucination by comparing with ground truth
    and evaluation dataset using TF-IDF similarity.

    Args:
        generated_answer: The generated answer to evaluate
        ground_truth: The correct ground truth answer
        evaluation_dataset: List of reference examples

    Returns:
        Dictionary containing evaluation results and analysis
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

    try:
        # Prepare data for TF-IDF analysis
        evaluation_labels = [entry["answer"].strip().lower() for entry in evaluation_dataset]
        all_answers = [generated_answer.strip().lower()] + evaluation_labels

        # Create TF-IDF embeddings
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_answers)

        # Calculate similarities
        generated_vec = tfidf_matrix[0].toarray()
        label_vecs = tfidf_matrix[1:].toarray()

        cosine_similarities = np.dot(label_vecs, generated_vec.T).flatten() / (
            np.linalg.norm(label_vecs, axis=1) * np.linalg.norm(generated_vec)
        )

        # Find best match
        best_match_index = np.argmax(cosine_similarities)
        highest_similarity = cosine_similarities[best_match_index]
        best_match = evaluation_labels[best_match_index]

        # Determine if answer is faithful based on similarity threshold
        normalized_answer = best_match if highest_similarity > 0.75 else "Not Faithful"
        is_correct = normalized_answer.lower() == ground_truth.lower()
        score = 1.0 if is_correct else 0.0

        # Generate analysis
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

    except Exception as e:
        logging.error(f"Error in hallucination evaluation: {str(e)}")
        return {
            "answer": "Error",
            "analysis": f"Evaluation failed: {str(e)}",
            "is_correct": False,
            "score": 0.0,
        }

def transform_eval_dataset_to_eval_json(eval_dataset):
    """Transform evaluation dataset to JSON format."""
    eval_json = {
        "queries": {},
        "responses": {}
    }
    for idx, entry in enumerate(eval_dataset, start=1):
        query_key = f"query{idx}"
        eval_json["queries"][query_key] = entry['query']
        eval_json["responses"][query_key] = entry['answer']
    return eval_json

def evaluate_responses(
    self,
    responses: List[str],
    ref_responses: Optional[List[str]] = None,
    evaluator: Optional[Any] = None,
    evaluation_dataset: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """Evaluate responses against reference answers."""
    try:
        eval_results = []
        for i, response in enumerate(responses):
            if not response:
                eval_results.append({"score": 0.0})
                continue

            result = {"response": response}

            if ref_responses and i < len(ref_responses):
                ref = ref_responses[i]
                score = self._calculate_similarity(response, ref)
                result["reference"] = ref
            else:
                score = self._heuristic_score(response)

            result["score"] = min(max(score, 0.0), 1.0)

            if evaluation_dataset and i < len(evaluation_dataset):
                result.update(evaluation_dataset[i])

            eval_results.append(result)

        return eval_results

    except Exception as e:
        logging.error(f"Error evaluating responses: {str(e)}")
        return [{"score": 0.0} for _ in responses]

def calculate_mean_score(
    self,
    eval_results: List[Dict[str, Any]],
    params: Optional[Set[str]] = None,
    evaluator: Optional[Any] = None,
    **kwargs
) -> Dict[str, float]:
    """Calculate mean scores for evaluation results."""
    if not eval_results:
        return {}

    try:
        if evaluator and hasattr(evaluator, 'calculate_mean_score'):
            return evaluator.calculate_mean_score(eval_results)

        mean_scores = {}
        for result in eval_results:
            result_params = result.get("params", {})
            param_dict = {
                k: result_params.get(k)
                for k in (params or result_params.keys())
            }
            score = result.get("score", 0.0)

            param_key = str(tuple(sorted((k, param_dict[k]) for k in param_dict)))

            if param_key not in mean_scores:
                mean_scores[param_key] = []
            mean_scores[param_key].append(score)

        return {
            k: float(mean(v)) if v else 0.0
            for k, v in mean_scores.items()
        }

    except Exception as e:
        logging.error(f"Error calculating mean score: {str(e)}")
        return {}
