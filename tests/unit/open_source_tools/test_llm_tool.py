"""tests for llm tool."""

from unittest.mock import MagicMock

import pytest

from portia.model import Message
from portia.open_source_tools.llm_tool import LLMTool, LLMToolSchema
from portia.tool import ToolRunContext


@pytest.fixture
def mock_llm_tool() -> LLMTool:
    """Fixture to create an instance of LLMTool."""
    return LLMTool(id="test_tool", name="Test LLM Tool")


def test_llm_tool_plan_run(
    mock_llm_tool: LLMTool,
    mock_tool_run_context: ToolRunContext,
    mock_model: MagicMock,
) -> None:
    """Test that LLMTool runs successfully and returns a response."""
    # Setup mock responses
    mock_model.get_response.return_value = Message(role="user", content="Test response content")
    # Define task input
    task = "What is the capital of France?"

    # Run the tool
    result = mock_llm_tool.run(mock_tool_run_context, task)

    mock_model.get_response.assert_called_once_with(
        [Message(role="user", content=mock_llm_tool.prompt), Message(role="user", content=task)],
    )

    # Assert the result is the expected response
    assert result == "Test response content"


def test_llm_tool_schema_valid_input() -> None:
    """Test that the LLMToolSchema correctly validates the input."""
    schema_data = {"task": "Solve a math problem"}
    schema = LLMToolSchema(**schema_data)

    assert schema.task == "Solve a math problem"


def test_llm_tool_schema_missing_task() -> None:
    """Test that LLMToolSchema raises an error if 'task' is missing."""
    with pytest.raises(ValueError):  # noqa: PT011
        LLMToolSchema()  # type: ignore  # noqa: PGH003


def test_llm_tool_initialization(mock_llm_tool: LLMTool) -> None:
    """Test that LLMTool is correctly initialized."""
    assert mock_llm_tool.id == "test_tool"
    assert mock_llm_tool.name == "Test LLM Tool"


def test_llm_tool_run_with_context(
    mock_llm_tool: LLMTool,
    mock_tool_run_context: ToolRunContext,
    mock_model: MagicMock,
) -> None:
    """Test that LLMTool runs successfully when a context is provided."""
    # Setup mock responses
    mock_model.get_response.return_value = Message(role="user", content="Test response content")

    # Define task and context
    mock_llm_tool.tool_context = "Context for task"
    task = "What is the capital of France?"

    # Run the tool
    result = mock_llm_tool.run(mock_tool_run_context, task)

    # Verify that the Model's get_response method is called
    called_with = mock_model.get_response.call_args_list[0].args[0]
    assert len(called_with) == 2
    assert isinstance(called_with[0], Message)
    assert isinstance(called_with[1], Message)
    assert mock_llm_tool.tool_context in called_with[1].content
    assert task in called_with[1].content
    # Assert the result is the expected response
    assert result == "Test response content"
