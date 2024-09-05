import pytest

from nomadic.model import OpenAIModel


def test_model_initialization():
    test_model = OpenAIModel(
        key_name="test-openai-model", api_keys={"OPENAI_API_KEY": "123456789"}
    )


if __name__ == "__main__":
    pytest.main()
