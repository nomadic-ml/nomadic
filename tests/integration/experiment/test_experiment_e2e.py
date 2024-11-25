import json
import nest_asyncio
import pytest
import requests
import os

from llama_index.core.evaluation import BatchEvalRunner
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.embeddings.openai import OpenAIEmbedding

from nomadic.experiment import Experiment
from nomadic.experiment.rag import obtain_rag_inputs, run_rag_pipeline
from nomadic.model import OpenAIModel
from nomadic.tuner import tune

from dotenv import dotenv_values

dotenv_values = dotenv_values(".env.dev")

nest_asyncio.apply()


def test_simple_openai_experiment():
    # Run a generic experiment
    experiment = Experiment(
        name="Sample_Nomadic_Experiment",
        model=OpenAIModel(api_keys={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]}),
        params={"temperature", "max_tokens"},
        evaluation_dataset=json.loads(
            requests.get(
                "https://dl.dropboxusercontent.com/scl/fi/y1tpv7kahcy5tfdh243rr/knowtex_llama2_prompts_example.json?rlkey=vf5y3g83r8n2xiwgtbqti01rk&e=1&st=68ceo8nr&dl=0"
            ).content
        ),
        evaluator=SemanticSimilarityEvaluator(embed_model=OpenAIEmbedding()),
    )

    expeirment_result = experiment.run(
        param_dict={
            "temperature": tune.choice([0.1, 0.9]),
            "max_tokens": tune.choice([250, 500]),
        }
    )

    # Our search space is 2 by 2 hyperparameter values, thereby yielding 4 results
    assert len(expeirment_result.run_results) == 4


def test_advanced_prompt_tuning_experiment():
    # Run advanced prompt tuning experiment
    # Initialize the sample evaluation dataset

    ## Initialize the prompt template
    prompt_template = """
    "Describe the capital city of the country Zephyria, including its most famous landmark and the year it was founded."
    """

    # Initialize the evaluator
    evaluator = {
        "method": "custom_evaluate",
        "evaluation_metrics": [
            {"metric": "Accuracy", "weight": 0.9},
            {"metric": "Simplicity", "weight": 0.1},
        ],
    }

    # Define search space
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
        model=OpenAIModel(
            model="gpt-4o", api_keys={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]}
        ),
        evaluator=evaluator,
        search_method="grid",
        enable_logging=False,
        use_flaml_library=False,
        fixed_param_dict={
        "prompt_tuning_topic": "hallucination-detection"
    }
    )

    experiment_result = experiment.run(
        param_dict={
            "temperature": temperature_search_space,
            "max_tokens": max_tokens_search_space,
            "prompt_tuning_approach": prompt_tuning_approach,
            "prompt_tuning_complexity": prompt_tuning_complexity,
        }
    )

    # Given 2*2*3*2=24 possible HP combinations
    assert len(experiment_result.run_results) == 24


def test_rag_experiment_only_obj_function():
    # Define search space
    top_k_search_space = tune.choice([1, 2])
    model_search_space = tune.choice(["gpt-3.5-turbo", "gpt-4o"])

    eval_json = {
        "queries": {
            "capital_city_question_1": "Describe the capital city of the country Zephyria, including its most famous landmark and the year it was founded.",
            "capital_city_question_2": "What is the name of the capital city of Zephyria, and what are some key historical events that took place there?",
            "capital_city_question_3": "Provide an overview of Zephyria's capital city, including its population size, economic significance, and major cultural institutions.",
        },
        "responses": {
            "capital_city_question_1": "As Zephyria is a fictional country, it doesn't have a real capital. However, in its fictional narrative, the capital city is Zephyros, which is said to have been founded in 1024 AD. The city is renowned for the Skyward Tower, a mythical landmark that is central to Zephyria's lore.",
            "capital_city_question_2": "Since Zephyria is a fictional country, it doesn’t have an actual capital city. But in the stories and lore surrounding Zephyria, Zephyros is considered the capital. Significant fictional events include the Great Treaty of 1456 and the construction of the Skyward Tower in 1602, both pivotal moments in Zephyros’ imagined history.",
            "capital_city_question_3": "Zephyria, being a fictional country, does not have a real capital. However, within its fictional context, Zephyros serves as the capital city, portrayed with a population of around 3 million. It is depicted as the economic and cultural heart of Zephyria, featuring legendary institutions like the Zephyros Museum of Art and the National Opera House, which are central to the country's fictional cultural narrative.",
        },
    }
    pdf_url = "https://www.dropbox.com/scl/fi/7dwj3g3fz2xqt7xt642a0/fakecountries-fandom-com-wiki-Zephyria.pdf?rlkey=7g93kdtb8zx775offoiaf89lo&st=pkces2nn&dl=1"

    docs, eval_qs, ref_response_strs = obtain_rag_inputs(
        pdf_url=pdf_url, eval_json=eval_json
    )
    experiment = Experiment(
        name="my rag experiment",
        param_fn=run_rag_pipeline,
        params={"top_k", "model_name"},
        fixed_param_dict={
            "docs": docs,
            "eval_qs": eval_qs[:10],
            "ref_response_strs": ref_response_strs[:10],
        },
        use_flaml_library=True
    )

    experiment_result_rag = experiment.run(
        param_dict={"top_k": top_k_search_space, "model_name": model_search_space}
    )
    # Given 2*2=4 possible HP combinations
    assert len(experiment_result_rag.run_results) == 4
