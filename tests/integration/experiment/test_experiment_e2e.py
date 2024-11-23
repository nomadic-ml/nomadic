import json
import nest_asyncio
import pytest
import os
from unittest.mock import Mock, patch

from llama_index.core.evaluation import BatchEvalRunner
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.embeddings.openai import OpenAIEmbedding

from nomadic.experiment import Experiment
from nomadic.experiment.rag import obtain_rag_inputs, run_rag_pipeline
from nomadic.model import OpenAIModel
from nomadic.tuner import tune

nest_asyncio.apply()


@patch("requests.get")
def test_simple_openai_experiment(mock_get):
    mock_get.return_value.content = json.dumps(
        [{"Instruction": "Test instruction", "Context": "Test context", "Answer": "Test answer"}]
    )

    experiment = Experiment(
        name="Sample_Nomadic_Experiment",
        model=Mock(OpenAIModel),
        params={"temperature", "max_tokens"},
        evaluation_dataset=json.loads(mock_get.return_value.content),
        evaluator=Mock(SemanticSimilarityEvaluator),
    )

    experiment_result = experiment.run(
        param_dict={
            "temperature": tune.choice([0.1, 0.9]),
            "max_tokens": tune.choice([250, 500]),
        }
    )

    assert experiment_result is not None
    assert hasattr(experiment_result, "run_results")


def test_advanced_prompt_tuning_experiment():
    prompt_template = """
    "Describe the capital city of the country Zephyria, including its most famous landmark and the year it was founded."
    """

    temperature_search_space = tune.choice([0.1, 0.9])
    max_tokens_search_space = tune.choice([50, 100])
    prompt_tuning_approach = tune.choice(["zero-shot", "few-shot", "chain-of-thought"])
    prompt_tuning_complexity = tune.choice(["simple", "complex"])

    experiment = Experiment(
        params={
            "temperature",
            "max_tokens",
            "prompt_tuning_approach",
            "prompt_tuning_complexity",
        },
        user_prompt_request=prompt_template,
        model=Mock(OpenAIModel),
        evaluator=Mock(),
        search_method="grid",
        enable_logging=False,
        use_flaml_library=False,
        fixed_param_dict={"prompt_tuning_topic": "hallucination-detection"},
    )

    experiment_result = experiment.run(
        param_dict={
            "temperature": temperature_search_space,
            "max_tokens": max_tokens_search_space,
            "prompt_tuning_approach": prompt_tuning_approach,
            "prompt_tuning_complexity": prompt_tuning_complexity,
        }
    )

    assert experiment_result is not None
    assert hasattr(experiment_result, "run_results")


@patch("nomadic.experiment.rag.obtain_rag_inputs")
def test_rag_experiment_only_obj_function(mock_obtain_rag_inputs):
    mock_docs = ["doc1", "doc2"]
    mock_eval_qs = ["query1", "query2", "query3"]
    mock_ref_responses = ["response1", "response2", "response3"]

    mock_obtain_rag_inputs.return_value = (mock_docs, mock_eval_qs, mock_ref_responses)

    top_k_search_space = tune.choice([1, 2])
    model_search_space = tune.choice(["gpt-3.5-turbo", "gpt-4o"])

    experiment = Experiment(
        name="my rag experiment",
        param_fn=run_rag_pipeline,
        params={"top_k", "model_name"},
        fixed_param_dict={
            "docs": mock_docs,
            "eval_qs": mock_eval_qs[:10],
            "ref_response_strs": mock_ref_responses[:10],
        },
    )

    experiment_result_rag = experiment.run(
        param_dict={"top_k": top_k_search_space, "model_name": model_search_space}
    )

    assert experiment_result_rag is not None
    assert hasattr(experiment_result_rag, "run_results")
