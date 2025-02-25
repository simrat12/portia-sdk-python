"""wrapper tests."""

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from pydantic import SecretStr

from portia.config import LLMModel
from portia.llm_wrapper import LLMWrapper
from portia.plan import Plan

MODELS = [
    LLMModel.GPT_4_O_MINI,
    LLMModel.MISTRAL_LARGE_LATEST,
    LLMModel.CLAUDE_3_OPUS,
]


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.flaky(reruns=3)  # MistralAI is a little flaky on the to_instructor call
def test_wrapper_methods(model_name: LLMModel) -> None:
    """Test we can generate wrappers for important providers."""
    wrapper = LLMWrapper(model_name=model_name, api_key=SecretStr("test123"))
    # check we don't get errors
    wrapper.to_instructor(
        Plan,
        [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "test"},
        ],
    )
    model = wrapper.to_langchain()

    if isinstance(model, (ChatMistralAI, ChatAnthropic)):
        # MistralAI and Anthropic get_name() method doesn't give the model name unfortunately
        assert model.model == model_name.value
    else:
        assert model.get_name() == model_name.value
