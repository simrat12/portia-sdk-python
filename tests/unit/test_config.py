"""Tests for portia classes."""

from unittest.mock import MagicMock, Mock

import pytest
from pydantic import SecretStr

from portia.config import (
    EXECUTION_MODEL_KEY,
    FEATURE_FLAG_AGENT_MEMORY_ENABLED,
    PLANNING_MODEL_KEY,
    Config,
    ExecutionAgentType,
    LLMModel,
    LLMProvider,
    LogLevel,
    PlanningAgentType,
    StorageClass,
)
from portia.errors import ConfigNotFoundError, InvalidConfigError
from portia.model import (
    AzureOpenAIGenerativeModel,
    GenerativeModel,
    LangChainGenerativeModel,
)
from tests.utils import get_test_config


def test_from_default() -> None:
    """Test from default."""
    c = Config.from_default(
        default_log_level=LogLevel.CRITICAL,
        openai_api_key=SecretStr("123"),
    )
    assert c.default_log_level == LogLevel.CRITICAL


def test_set_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test setting keys."""
    monkeypatch.setenv("PORTIA_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")
    c = Config.from_default(default_log_level=LogLevel.CRITICAL)
    assert c.portia_api_key == SecretStr("test-key")
    assert c.openai_api_key == SecretStr("test-openai-key")
    assert c.anthropic_api_key == SecretStr("test-anthropic-key")
    assert c.mistralai_api_key == SecretStr("test-mistral-key")


def test_set_with_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test setting keys as string."""
    monkeypatch.setenv("PORTIA_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")
    # storage
    c = Config.from_default(storage_class="MEMORY")
    assert c.storage_class == StorageClass.MEMORY

    c = Config.from_default(storage_class="DISK", storage_dir="/test")
    assert c.storage_class == StorageClass.DISK
    assert c.storage_dir == "/test"

    # Need to specify storage_dir if using DISK
    with pytest.raises(InvalidConfigError):
        c = Config.from_default(storage_class="DISK")

    with pytest.raises(InvalidConfigError):
        c = Config.from_default(storage_class="OTHER")

    with pytest.raises(InvalidConfigError):
        c = Config.from_default(storage_class=123)

    # log level
    c = Config.from_default(default_log_level="CRITICAL")
    assert c.default_log_level == LogLevel.CRITICAL
    with pytest.raises(InvalidConfigError):
        c = Config.from_default(default_log_level="some level")

    # execution_agent_type
    c = Config.from_default(execution_agent_type="default")
    assert c.execution_agent_type == ExecutionAgentType.DEFAULT
    with pytest.raises(InvalidConfigError):
        c = Config.from_default(execution_agent_type="my agent")

    # Large output threshold value
    c = Config.from_default(
        large_output_threshold_tokens=100,
        feature_flags={
            FEATURE_FLAG_AGENT_MEMORY_ENABLED: True,
        },
    )
    assert c.large_output_threshold_tokens == 100
    assert c.exceeds_output_threshold("Test " * 1000)
    c = Config.from_default(
        large_output_threshold_tokens=100,
        feature_flags={
            FEATURE_FLAG_AGENT_MEMORY_ENABLED: False,
        },
    )
    assert c.large_output_threshold_tokens == 100
    assert not c.exceeds_output_threshold("Test " * 1000)


def test_set_llms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test setting LLM models."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")

    # Models can be set individually
    c = Config.from_default(
        planning_model_name=LLMModel.GPT_4_O,
        execution_model_name=LLMModel.GPT_4_O_MINI,
    )
    assert c.model(PLANNING_MODEL_KEY) == LLMModel.GPT_4_O
    assert c.model(EXECUTION_MODEL_KEY) == LLMModel.GPT_4_O_MINI

    # llm_model_name sets all models
    c = Config.from_default(llm_model_name="mistral_large")
    assert c.model(PLANNING_MODEL_KEY) == LLMModel.MISTRAL_LARGE
    assert c.model(EXECUTION_MODEL_KEY) == LLMModel.MISTRAL_LARGE

    # llm_provider sets default model for all providers
    c = Config.from_default(llm_provider="mistralai")
    assert c.model(PLANNING_MODEL_KEY) == LLMModel.MISTRAL_LARGE
    assert c.model(EXECUTION_MODEL_KEY) == LLMModel.MISTRAL_LARGE

    # With nothing specified, it chooses a model we have API keys for
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")
    c = Config.from_default()
    assert c.model(PLANNING_MODEL_KEY) == LLMModel.MISTRAL_LARGE
    assert c.model(EXECUTION_MODEL_KEY) == LLMModel.MISTRAL_LARGE

    # With all API key set, correct default models are chosen
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    c = Config.from_default()
    assert c.model(PLANNING_MODEL_KEY) == LLMModel.O_3_MINI
    assert c.model(EXECUTION_MODEL_KEY) == LLMModel.GPT_4_O

    # No api key for provider model
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "")
    for provider in [
        LLMProvider.OPENAI,
        LLMProvider.ANTHROPIC,
        LLMProvider.MISTRALAI,
        LLMProvider.GOOGLE_GENERATIVE_AI,
        LLMProvider.AZURE_OPENAI,
    ]:
        with pytest.raises(InvalidConfigError):
            Config.from_default(
                storage_class=StorageClass.MEMORY,
                llm_provider=provider,
                execution_agent_type=ExecutionAgentType.DEFAULT,
                planning_agent_type=PlanningAgentType.DEFAULT,
            )

    # Wrong api key for provider model
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    with pytest.raises(InvalidConfigError):
        Config.from_default(
            storage_class=StorageClass.MEMORY,
            llm_model_name=LLMModel.MISTRAL_LARGE,
            execution_agent_type=ExecutionAgentType.DEFAULT,
            planning_agent_type=PlanningAgentType.DEFAULT,
        )

    # Unrecognised providers error
    with pytest.raises(InvalidConfigError):
        c = Config.from_default(llm_provider="personal", llm_model_name="other-model")


def test_resolve_model_azure() -> None:
    """Test resolve model for Azure OpenAI."""
    c = Config.from_default(
        llm_provider=LLMProvider.AZURE_OPENAI,
        azure_openai_endpoint="http://test-azure-openai-endpoint",
        azure_openai_api_key="test-azure-openai-api-key",
    )
    assert isinstance(c.resolve_model(PLANNING_MODEL_KEY), AzureOpenAIGenerativeModel)


def test_resolve_langchain_model() -> None:
    """Test resolve langchain model."""
    conf = get_test_config(
        custom_models={
            PLANNING_MODEL_KEY: LangChainGenerativeModel(client=MagicMock(), model_name="test"),
        },
    )
    assert isinstance(conf.resolve_langchain_model(PLANNING_MODEL_KEY), LangChainGenerativeModel)


def test_resolve_langchain_model_error() -> None:
    """Test resolve langchain model raises TypeError if model is not a LangChainGenerativeModel."""
    conf = get_test_config(
        custom_models={
            PLANNING_MODEL_KEY: Mock(spec=GenerativeModel),
        },
    )
    with pytest.raises(TypeError, match="A LangChainGenerativeModel is required"):
        conf.resolve_langchain_model(PLANNING_MODEL_KEY)


def test_custom_models() -> None:
    """Test custom models."""
    c = Config.from_default(
        custom_models={
            PLANNING_MODEL_KEY: LangChainGenerativeModel(
                client=MagicMock(),
                model_name="gpt-4o",
            ),
        },
        openai_api_key=SecretStr("test-openai-key"),
    )
    resolved_model = c.resolve_model(PLANNING_MODEL_KEY)
    assert isinstance(resolved_model, LangChainGenerativeModel)
    assert resolved_model.model_name == "gpt-4o"


def test_getters() -> None:
    """Test getters work."""
    c = Config.from_default(
        openai_api_key=SecretStr("123"),
    )

    assert c.has_api_key("openai_api_key")

    with pytest.raises(ConfigNotFoundError):
        c.must_get("not real", str)

    c = Config.from_default(
        openai_api_key=SecretStr("123"),
        portia_api_key=SecretStr("123"),
        anthropic_api_key=SecretStr(""),
        portia_api_endpoint="",
        portia_dashboard_url="",
    )
    with pytest.raises(InvalidConfigError):
        c.must_get("portia_api_key", int)

    with pytest.raises(InvalidConfigError):
        c.must_get("portia_api_endpoint", str)

    with pytest.raises(InvalidConfigError):
        c.must_get("portia_dashboard_url", str)

    # no Portia API Key
    with pytest.raises(InvalidConfigError):
        Config.from_default(
            storage_class=StorageClass.CLOUD,
            portia_api_key=SecretStr(""),
            execution_agent_type=ExecutionAgentType.DEFAULT,
            planning_agent_type=PlanningAgentType.DEFAULT,
        )


def test_azure_openai_requires_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Azure OpenAI requires endpoint."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-azure-openai-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "")

    # Without endpoint set, it errors
    with pytest.raises(InvalidConfigError):
        Config.from_default(llm_provider=LLMProvider.AZURE_OPENAI)

    # With endpoint set, it works
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "test-azure-openai-endpoint")
    c = Config.from_default(llm_provider=LLMProvider.AZURE_OPENAI)
    assert c.llm_provider == LLMProvider.AZURE_OPENAI
    assert c.model(PLANNING_MODEL_KEY).provider() == LLMProvider.AZURE_OPENAI

    # Also works with passing parameters to constructor
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "")
    c = Config.from_default(
        llm_provider=LLMProvider.AZURE_OPENAI,
        azure_openai_endpoint="test-azure-openai-endpoint",
        azure_openai_api_key="test-azure-openai-api-key",
    )
    assert c.llm_provider == LLMProvider.AZURE_OPENAI


@pytest.mark.parametrize("model", list(LLMModel))
def test_all_models_have_provider(model: LLMModel) -> None:
    """Test all models have a provider."""
    assert model.provider() is not None


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("gpt-4o", LLMModel.GPT_4_O),
        ("openai/gpt-4o", LLMModel.GPT_4_O),
        ("azure_openai/gpt-4o", LLMModel.AZURE_GPT_4_O),
        ("claude-3-5-haiku-latest", LLMModel.CLAUDE_3_5_HAIKU),
        ("mistral-large-latest", LLMModel.MISTRAL_LARGE),
        ("gemini-2.0-flash", LLMModel.GEMINI_2_0_FLASH),
    ],
)
def test_llm_model_instantiate_from_string(model_name: str, expected: LLMModel) -> None:
    """Test LLM model from string."""
    model = LLMModel(model_name)
    assert model == expected


def test_llm_model_instantiate_from_string_missing() -> None:
    """Test LLM model from string missing."""
    with pytest.raises(ValueError, match="Invalid LLM model"):
        LLMModel("not-a-model")


PROVIDER_ENV_VARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "MISTRAL_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
]


def clear_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear env vars for the Provider APIs."""
    for env_var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


@pytest.mark.parametrize(
    ("env_vars", "provider"),
    [
        ({"OPENAI_API_KEY": "test-openai-api-key"}, LLMProvider.OPENAI),
        ({"ANTHROPIC_API_KEY": "test-anthropic-api-key"}, LLMProvider.ANTHROPIC),
        ({"MISTRAL_API_KEY": "test-mistral-api-key"}, LLMProvider.MISTRALAI),
        ({"GOOGLE_API_KEY": "test-google-api-key"}, LLMProvider.GOOGLE_GENERATIVE_AI),
        (
            {
                "AZURE_OPENAI_API_KEY": "test-azure-openai-api-key",
                "AZURE_OPENAI_ENDPOINT": "test-azure-openai-endpoint",
            },
            LLMProvider.AZURE_OPENAI,
        ),
    ],
)
def test_llm_provider_default_from_api_keys_env_vars(
    env_vars: dict[str, str],
    provider: LLMProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test LLM provider default from API keys env vars."""
    clear_env_vars(monkeypatch)
    for env_var_name, env_var_value in env_vars.items():
        monkeypatch.setenv(env_var_name, env_var_value)

    c = Config.from_default()
    assert c.llm_provider == provider


@pytest.mark.parametrize(
    ("config_kwargs", "provider"),
    [
        ({"openai_api_key": "test-openai-api-key"}, LLMProvider.OPENAI),
        ({"anthropic_api_key": "test-anthropic-api-key"}, LLMProvider.ANTHROPIC),
        ({"mistralai_api_key": "test-mistral-api-key"}, LLMProvider.MISTRALAI),
        ({"google_api_key": "test-google-api-key"}, LLMProvider.GOOGLE_GENERATIVE_AI),
        (
            {
                "azure_openai_api_key": "test-azure-openai-api-key",
                "azure_openai_endpoint": "test-azure-openai-endpoint",
            },
            LLMProvider.AZURE_OPENAI,
        ),
    ],
)
def test_llm_provider_default_from_api_keys_config_kwargs(
    config_kwargs: dict[str, str],
    provider: LLMProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test LLM provider default from API keys config kwargs."""
    clear_env_vars(monkeypatch)
    c = Config.from_default(**config_kwargs)
    assert c.llm_provider == provider
