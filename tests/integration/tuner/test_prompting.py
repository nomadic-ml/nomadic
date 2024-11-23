import os
import json
import requests

from nomadic.experiment import Experiment
from nomadic.model import OpenAIModel
from nomadic.tuner import tune

from dotenv import dotenv_values

dotenv_values = dotenv_values(".env.dev")

def test_prompt_tuning_experiment():
    prompt_template = """
        You are an AI assistant specialized in detecting hallucinations in text responses. Your task is to analyze the given query, context, and response, and determine if the response contains any hallucinations or unfaithful information.
        Instructions:
        1. Carefully read the query, context, and response.
        2. Compare the information in the response to the provided context.
        3. Identify any statements in the response that are not supported by or contradict the context.
        4. Determine if the response is faithful to the query and context, or if it contains hallucinations.
        Provide only your judgment as one of the following (just one word):
        - "Faithful"
        - "Not faithful"
        - "Refusal"
        Do not provide any additional explanation or analysis.
        Query: [QUERY]
        Context: [CONTEXT]
        Response to evaluate: [RESPONSE]
        Your one-word judgment:
    """

    temperature_search_space = tune.choice([0.1, 0.9])
    max_tokens_search_space = tune.choice([100])
    prompt_tuning_approach = tune.choice(["chain-of-thought"])
    prompt_tuning_complexity = tune.choice(["complex"])
    prompt_tuning_focus = tune.choice(["british-english-adherence"])
    prompt_tuning_topic = tune.choice(["hallucination-detection"])

    experiment = Experiment(
        params={"temperature", "max_tokens", "prompt_approach", "prompt_complexity", "prompt_tuning_focus", "prompt_tuning_topic"},
        user_prompt_request=prompt_template,
        model=OpenAIModel(model="gpt-4o-mini", api_keys={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]}),
        evaluator={"method": "custom_evaluate_hallucination"},
        search_method="grid",
        enable_logging=False,
        use_flaml_library=False,
        name="Hallucination Detection Experiment",
        evaluation_dataset=json.loads(
            requests.get(
                "https://dl.dropboxusercontent.com/scl/fi/5n516glrcg3ng0xinhkca/prompt_tuning_hallucination_detection_example.json?rlkey=1ugzbkvqczw4ko5gn2rusrphf&dl=0"
            ).content
        ),
        num_simulations=3,
    )

    experiment_result = experiment.run(
        param_dict={
            "temperature": temperature_search_space,
            "max_tokens": max_tokens_search_space,
            "prompt_tuning_approach": prompt_tuning_approach,
            "prompt_tuning_complexity": prompt_tuning_complexity,
            "prompt_tuning_focus": prompt_tuning_focus,
            "prompt_tuning_topic": prompt_tuning_topic,
        }
    )

    assert experiment_result is not None
    assert hasattr(experiment_result, "run_results")
    assert len(experiment_result.run_results) >= 0
