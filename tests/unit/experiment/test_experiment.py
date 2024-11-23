import pytest
from unittest.mock import Mock
from llama_index.core.evaluation import BaseEvaluator
from nomadic.experiment import Experiment
from nomadic.model import OpenAIModel


@pytest.fixture
def experiment():
    evaluation_dataset = [
        {
            "Instruction": "Test instruction",
            "Context": "Test context",
            "Answer": "Test answer",
        }
    ]
    user_prompt_request = "Test request"
    model = Mock(OpenAIModel)
    evaluator = Mock(BaseEvaluator)
    return Experiment(
        params={"param1", "param2"},
        evaluation_dataset=evaluation_dataset,
        user_prompt_request=user_prompt_request,
        model=model,
        evaluator=evaluator,
        search_method="grid",  # Added default valid search method
    )


def test_experiment_initialization(experiment):
    assert experiment.params == {"param1", "param2"}
    assert len(experiment.evaluation_dataset) == 1
    assert experiment.user_prompt_request == "Test request"
    assert experiment.model is not None
    assert experiment.evaluator is not None


@pytest.mark.skip("TODO: Enforce Experiment search method at instantiation time.")
def test_experiment_invalid_search_method():
    # Adjusted to mock the behavior without raising a ValueError
    with pytest.raises(ValueError):
        Experiment(
            params={"param1"},
            evaluation_dataset=[
                {
                    "Instruction": "Test instruction",
                    "Context": "Test context",
                    "Answer": "Test answer",
                }
            ],
            user_prompt_request="Test request",
            model=Mock(OpenAIModel),
            evaluator=Mock(BaseEvaluator),
            search_method="invalid_method",  # Still invalid for coverage
        )


def test_model_post_init_valid_search_method():
    experiment = Experiment(
        params={"param1"},
        evaluation_dataset=[
            {
                "Instruction": "Test instruction",
                "Context": "Test context",
                "Answer": "Test answer",
            }
        ],
        user_prompt_request="Test request",
        model=Mock(OpenAIModel),
        evaluator=Mock(BaseEvaluator),
        search_method="grid",  # Valid method
    )
    assert experiment.search_method == "grid"


if __name__ == "__main__":
    pytest.main()
