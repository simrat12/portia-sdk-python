"""Tests for portia classes."""

import tempfile
import threading
import time
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest
from pydantic import HttpUrl, SecretStr

from portia.clarification import (
    ActionClarification,
    InputClarification,
    ValueConfirmationClarification,
)
from portia.config import (
    DEFAULT_MODEL_KEY,
    FEATURE_FLAG_AGENT_MEMORY_ENABLED,
    PLANNING_MODEL_KEY,
    Config,
    StorageClass,
)
from portia.errors import InvalidPlanRunStateError, PlanError, PlanRunNotFoundError
from portia.execution_agents.output import AgentMemoryOutput, LocalOutput
from portia.introspection_agents.introspection_agent import (
    PreStepIntrospection,
    PreStepIntrospectionOutcome,
)
from portia.model import LangChainGenerativeModel
from portia.open_source_tools.llm_tool import LLMTool
from portia.open_source_tools.registry import example_tool_registry, open_source_tool_registry
from portia.plan import Plan, PlanContext, ReadOnlyPlan, Step
from portia.plan_run import PlanRun, PlanRunOutputs, PlanRunState, PlanRunUUID, ReadOnlyPlanRun
from portia.planning_agents.base_planning_agent import StepsOrError
from portia.portia import ExecutionHooks, Portia
from portia.tool import Tool, ToolRunContext
from portia.tool_registry import ToolRegistry
from tests.utils import (
    AdditionTool,
    ClarificationTool,
    TestClarificationHandler,
    get_test_config,
    get_test_plan_run,
)


@pytest.fixture
def planning_model() -> MagicMock:
    """Fixture to create a mock planning model."""
    return MagicMock(spec=LangChainGenerativeModel)


@pytest.fixture
def default_model() -> MagicMock:
    """Fixture to create a mock default model."""
    return MagicMock(spec=LangChainGenerativeModel)


@pytest.fixture
def portia(planning_model: MagicMock, default_model: MagicMock) -> Portia:
    """Fixture to create a Portia instance for testing."""
    config = get_test_config(
        custom_models={
            PLANNING_MODEL_KEY: planning_model,
            DEFAULT_MODEL_KEY: default_model,
        },
    )
    tool_registry = ToolRegistry([AdditionTool(), ClarificationTool()])
    return Portia(config=config, tools=tool_registry)


@pytest.fixture
def portia_with_agent_memory(planning_model: MagicMock, default_model: MagicMock) -> Portia:
    """Fixture to create a Portia instance for testing."""
    config = get_test_config(
        # Set a small threshold value so all outputs are stored in agent memory
        feature_flags={FEATURE_FLAG_AGENT_MEMORY_ENABLED: True},
        large_output_threshold_tokens=3,
        custom_models={
            PLANNING_MODEL_KEY: planning_model,
            DEFAULT_MODEL_KEY: default_model,
        },
    )
    tool_registry = ToolRegistry([AdditionTool(), ClarificationTool()])
    return Portia(config=config, tools=tool_registry)


def test_portia_local_default_config_with_api_keys() -> None:
    """Test that the default config is used if no config is provided."""
    # Unset the portia API env that the portia doesn't try to use Portia Cloud
    with mock.patch.dict(
        "os.environ",
        {
            "PORTIA_API_KEY": "",
            "OPENAI_API_KEY": "123",
            "TAVILY_API_KEY": "123",
            "OPENWEATHERMAP_API_KEY": "123",
        },
    ):
        portia = Portia()
        assert portia.config == Config.from_default()
        assert len(portia.tool_registry.get_tools()) == len(open_source_tool_registry.get_tools())


def test_portia_local_default_config_without_api_keys() -> None:
    """Test that the default config when no API keys are provided."""
    # Unset the Tavily and weather API and check that these aren't included in
    # the default tool registry
    with mock.patch.dict(
        "os.environ",
        {
            "PORTIA_API_KEY": "",
            "OPENAI_API_KEY": "123",
            "TAVILY_API_KEY": "",
            "OPENWEATHERMAP_API_KEY": "",
        },
    ):
        portia = Portia()
        assert portia.config == Config.from_default()
        assert (
            len(portia.tool_registry.get_tools()) == len(open_source_tool_registry.get_tools()) - 2
        )


def test_portia_run_query(portia: Portia, planning_model: MagicMock) -> None:
    """Test running a query."""
    query = "example query"

    planning_model.get_structured_response.return_value = StepsOrError(
        steps=[],
        error=None,
    )

    plan_run = portia.run(query)

    assert plan_run.state == PlanRunState.COMPLETE


def test_portia_run_query_tool_list(planning_model: MagicMock) -> None:
    """Test running a query."""
    query = "example query"
    portia = Portia(
        config=get_test_config(
            custom_models={
                PLANNING_MODEL_KEY: planning_model,
            },
        ),
        tools=[AdditionTool(), ClarificationTool()],
    )

    planning_model.get_structured_response.return_value = StepsOrError(
        steps=[],
        error=None,
    )
    plan_run = portia.run(query)

    assert plan_run.state == PlanRunState.COMPLETE


def test_portia_run_query_disk_storage(planning_model: MagicMock) -> None:
    """Test running a query."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        query = "example query"
        config = Config.from_default(
            storage_class=StorageClass.DISK,
            openai_api_key=SecretStr("123"),
            storage_dir=tmp_dir,
            custom_models={
                PLANNING_MODEL_KEY: planning_model,
            },
        )
        tool_registry = ToolRegistry([AdditionTool(), ClarificationTool()])
        portia = Portia(config=config, tools=tool_registry)

        planning_model.get_structured_response.return_value = StepsOrError(steps=[], error=None)
        plan_run = portia.run(query)

        assert plan_run.state == PlanRunState.COMPLETE
        # Use Path to check for the files
        plan_files = list(Path(tmp_dir).glob("plan-*.json"))
        run_files = list(Path(tmp_dir).glob("prun-*.json"))

        assert len(plan_files) == 1
        assert len(run_files) == 1


def test_portia_generate_plan(portia: Portia, planning_model: MagicMock) -> None:
    """Test planning a query."""
    query = "example query"

    planning_model.get_structured_response.return_value = StepsOrError(steps=[], error=None)
    plan = portia.plan(query)

    assert plan.plan_context.query == query


def test_portia_generate_plan_error(portia: Portia, planning_model: MagicMock) -> None:
    """Test planning a query that returns an error."""
    query = "example query"

    planning_model.get_structured_response.return_value = StepsOrError(
        steps=[],
        error="could not plan",
    )
    with pytest.raises(PlanError):
        portia.plan(query)


def test_portia_generate_plan_with_tools(portia: Portia, planning_model: MagicMock) -> None:
    """Test planning a query."""
    query = "example query"

    planning_model.get_structured_response.return_value = StepsOrError(steps=[], error=None)
    plan = portia.plan(query, tools=["add_tool"])

    assert plan.plan_context.query == query
    assert plan.plan_context.tool_ids == ["add_tool"]


def test_portia_resume(portia: Portia, planning_model: MagicMock) -> None:
    """Test running a plan."""
    query = "example query"

    planning_model.get_structured_response.return_value = StepsOrError(steps=[], error=None)
    plan = portia.plan(query)
    plan_run = portia.create_plan_run(plan)
    plan_run = portia.resume(plan_run)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.plan_id == plan.id


def test_portia_resume_after_interruption(portia: Portia, planning_model: MagicMock) -> None:
    """Test resuming PlanRun after interruption."""
    query = "example query"

    planning_model.get_structured_response.return_value = StepsOrError(steps=[], error=None)
    plan_run = portia.run(query)

    # Simulate run being in progress
    plan_run.state = PlanRunState.IN_PROGRESS
    plan_run.current_step_index = 1
    plan_run = portia.resume(plan_run)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.current_step_index == 1


def test_portia_resume_edge_cases(portia: Portia, planning_model: MagicMock) -> None:
    """Test edge cases for execute."""
    with pytest.raises(ValueError):  # noqa: PT011
        portia.resume()

    query = "example query"
    planning_model.get_structured_response.return_value = StepsOrError(
        steps=[],
        error=None,
    )
    plan = portia.plan(query)
    plan_run = portia.create_plan_run(plan)

    # Simulate run being in progress
    plan_run.state = PlanRunState.IN_PROGRESS
    plan_run.current_step_index = 1
    plan_run = portia.resume(plan_run_id=plan_run.id)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.current_step_index == 1

    with pytest.raises(PlanRunNotFoundError):
        portia.resume(plan_run_id=PlanRunUUID())


def test_portia_run_invalid_state(portia: Portia, planning_model: MagicMock) -> None:
    """Test resuming PlanRun with an invalid state."""
    query = "example query"

    planning_model.get_structured_response.return_value = StepsOrError(steps=[], error=None)
    plan_run = portia.run(query)

    # Set invalid state
    plan_run.state = PlanRunState.COMPLETE

    with pytest.raises(InvalidPlanRunStateError):
        portia.resume(plan_run)


def test_portia_wait_for_ready(portia: Portia, planning_model: MagicMock) -> None:
    """Test wait for ready."""
    query = "example query"

    planning_model.get_structured_response.return_value = StepsOrError(
        steps=[Step(task="Example task", inputs=[], output="$output")],
        error=None,
    )
    plan_run = portia.run(query)

    plan_run.state = PlanRunState.FAILED
    with pytest.raises(InvalidPlanRunStateError):
        portia.wait_for_ready(plan_run)

    plan_run.state = PlanRunState.IN_PROGRESS
    plan_run = portia.wait_for_ready(plan_run)
    assert plan_run.state == PlanRunState.IN_PROGRESS

    def update_run_state() -> None:
        """Update the run state after sleeping."""
        time.sleep(1)  # Simulate some delay before state changes
        plan_run.state = PlanRunState.READY_TO_RESUME
        portia.storage.save_plan_run(plan_run)

    plan_run.state = PlanRunState.NEED_CLARIFICATION

    # Ensure current_step_index is set to a valid index
    plan_run.current_step_index = 0
    portia.storage.save_plan_run(plan_run)

    # start a thread to update in status
    update_thread = threading.Thread(target=update_run_state)
    update_thread.start()

    plan_run = portia.wait_for_ready(plan_run)
    assert plan_run.state == PlanRunState.READY_TO_RESUME


def test_portia_wait_for_ready_tool(portia: Portia) -> None:
    """Test wait for ready."""
    mock_call_count = MagicMock()
    mock_call_count.__iadd__ = (
        lambda self, other: setattr(self, "count", self.count + other) or self
    )
    mock_call_count.count = 0

    class ReadyTool(Tool):
        """Returns ready."""

        id: str = "ready_tool"
        name: str = "Ready Tool"
        description: str = "Returns a clarification"
        output_schema: tuple[str, str] = (
            "Clarification",
            "Clarification: The value of the Clarification",
        )

        def run(self, ctx: ToolRunContext, user_guidance: str) -> str:  # noqa: ARG002
            return "result"

        def ready(self, ctx: ToolRunContext) -> bool:  # noqa: ARG002
            mock_call_count.count += 1
            return mock_call_count.count == 3

    portia.tool_registry = ToolRegistry([ReadyTool()])
    step0 = Step(
        task="Do something",
        inputs=[],
        output="$ctx_0",
    )
    step1 = Step(
        task="Save Context",
        inputs=[],
        output="$ctx",
        tool_id="ready_tool",
    )
    plan = Plan(
        plan_context=PlanContext(
            query="run the tool",
            tool_ids=["ready_tool"],
        ),
        steps=[step0, step1],
    )
    unresolved_action = ActionClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="",
        action_url=HttpUrl("https://unresolved.step1.com"),
        step=1,
    )
    plan_run = PlanRun(
        plan_id=plan.id,
        current_step_index=1,
        state=PlanRunState.NEED_CLARIFICATION,
        outputs=PlanRunOutputs(
            clarifications=[
                ActionClarification(
                    plan_run_id=PlanRunUUID(),
                    user_guidance="",
                    action_url=HttpUrl("https://resolved.step0.com"),
                    resolved=True,
                    step=0,
                ),
                ActionClarification(
                    plan_run_id=PlanRunUUID(),
                    user_guidance="",
                    action_url=HttpUrl("https://resolved.step1.com"),
                    resolved=True,
                    step=1,
                ),
                unresolved_action,
            ],
        ),
    )
    portia.storage.save_plan(plan)
    portia.storage.save_plan_run(plan_run)
    assert plan_run.get_outstanding_clarifications() == [unresolved_action]
    plan_run = portia.wait_for_ready(plan_run)
    assert plan_run.state == PlanRunState.READY_TO_RESUME
    assert plan_run.get_outstanding_clarifications() == []
    for clarification in plan_run.outputs.clarifications:
        if clarification.step == 1:
            assert clarification.resolved
            assert clarification.response == "complete"


def test_get_clarifications_and_get_run_called_once(
    portia: Portia,
    planning_model: MagicMock,
) -> None:
    """Test that get_clarifications_for_step is called once after get_plan_run."""
    query = "example query"
    planning_model.get_structured_response.return_value = StepsOrError(
        steps=[Step(task="Example task", inputs=[], output="$output")],
        error=None,
    )
    plan_run = portia.run(query)

    # Set the run state to NEED_CLARIFICATION to ensure it goes through the wait logic
    plan_run.state = PlanRunState.NEED_CLARIFICATION
    plan_run.current_step_index = 0  # Set to a valid index

    # Mock the storage methods
    with (
        mock.patch.object(
            portia.storage,
            "get_plan_run",
            return_value=plan_run,
        ) as mock_get_plan_run,
        mock.patch.object(
            PlanRun,
            "get_clarifications_for_step",
            return_value=[],
        ) as mock_get_clarifications,
    ):
        # Call wait_for_ready
        portia.wait_for_ready(plan_run)

        # Assert that get_run was called once
        mock_get_plan_run.assert_called_once_with(plan_run.id)

        # Assert that get_clarifications_for_step was called once after get_run
        mock_get_clarifications.assert_called_once()


def test_portia_run_query_with_summary(portia: Portia, planning_model: MagicMock) -> None:
    """Test run_query sets both final output and summary correctly."""
    query = "What activities can I do in London based on weather?"

    # Mock planning_agent response
    weather_step = Step(
        task="Get weather in London",
        tool_id="add_tool",
        output="$weather",
    )
    activities_step = Step(
        task="Suggest activities based on weather",
        tool_id="add_tool",
        output="$activities",
    )
    planning_model.get_structured_response.return_value = StepsOrError(
        steps=[weather_step, activities_step],
        error=None,
    )

    # Mock agent responses
    weather_output = LocalOutput(value="Sunny and warm")
    activities_output = LocalOutput(value="Visit Hyde Park and have a picnic")
    expected_summary = "Weather is sunny and warm in London, visit to Hyde Park for a picnic"

    mock_step_agent = mock.MagicMock()
    mock_step_agent.execute_sync.side_effect = [weather_output, activities_output]

    mock_summarizer_agent = mock.MagicMock()
    mock_summarizer_agent.create_summary.side_effect = [expected_summary]

    with (
        mock.patch(
            "portia.portia.FinalOutputSummarizer",
            return_value=mock_summarizer_agent,
        ),
        mock.patch.object(portia, "_get_agent_for_step", return_value=mock_step_agent),
    ):
        plan_run = portia.run(query)

        # Verify run completed successfully
        assert plan_run.state == PlanRunState.COMPLETE

        # Verify step outputs were stored correctly
        assert plan_run.outputs.step_outputs["$weather"] == weather_output
        assert plan_run.outputs.step_outputs["$activities"] == activities_output

        # Verify final output and summary
        assert plan_run.outputs.final_output is not None
        assert plan_run.outputs.final_output.get_value() == activities_output.get_value()
        assert plan_run.outputs.final_output.get_summary() == expected_summary

        # Verify create_summary was called with correct args
        mock_summarizer_agent.create_summary.assert_called_once_with(
            plan=mock.ANY,
            plan_run=mock.ANY,
        )


def test_portia_sets_final_output_with_summary(portia: Portia) -> None:
    """Test that final output is set with correct summary."""
    (plan, plan_run) = get_test_plan_run()
    plan.steps = [
        Step(
            task="Get weather in London",
            output="$london_weather",
        ),
        Step(
            task="Suggest activities based on weather",
            output="$activities",
        ),
    ]

    plan_run.outputs.step_outputs = {
        "$london_weather": LocalOutput(value="Sunny and warm"),
        "$activities": LocalOutput(value="Visit Hyde Park and have a picnic"),
    }

    expected_summary = "Weather is sunny and warm in London, visit to Hyde Park for a picnic"
    mock_summarizer = mock.MagicMock()
    mock_summarizer.create_summary.side_effect = [expected_summary]

    with mock.patch(
        "portia.portia.FinalOutputSummarizer",
        return_value=mock_summarizer,
    ):
        last_step_output = LocalOutput(value="Visit Hyde Park and have a picnic")
        output = portia._get_final_output(plan, plan_run, last_step_output)  # noqa: SLF001

        # Verify the final output
        assert output is not None
        assert output.get_value() == "Visit Hyde Park and have a picnic"
        assert output.get_summary() == expected_summary

        # Verify create_summary was called with correct args
        mock_summarizer.create_summary.assert_called_once()
        call_args = mock_summarizer.create_summary.call_args[1]
        assert isinstance(call_args["plan"], ReadOnlyPlan)
        assert isinstance(call_args["plan_run"], ReadOnlyPlanRun)
        assert call_args["plan"].id == plan.id
        assert call_args["plan_run"].id == plan_run.id


def test_portia_run_query_with_memory(
    portia_with_agent_memory: Portia,
    planning_model: MagicMock,
) -> None:
    """Test run_query sets both final output and summary correctly."""
    query = "What activities can I do in London based on weather?"

    # Mock planning_agent response
    weather_step = Step(
        task="Get weather in London",
        tool_id="add_tool",
        output="$weather",
    )
    activities_step = Step(
        task="Suggest activities based on weather",
        tool_id="add_tool",
        output="$activities",
    )
    planning_model.get_structured_response.return_value = StepsOrError(
        steps=[weather_step, activities_step],
        error=None,
    )

    # Mock agent responses
    weather_summary = "sunny"
    weather_output = LocalOutput(value="Sunny and warm", summary=weather_summary)
    activities_summary = "picnic"
    activities_output = LocalOutput(
        value="Visit Hyde Park and have a picnic",
        summary=activities_summary,
    )
    expected_summary = "Weather is sunny and warm in London, visit to Hyde Park for a picnic"

    mock_step_agent = mock.MagicMock()
    mock_step_agent.execute_sync.side_effect = [weather_output, activities_output]

    mock_summarizer_agent = mock.MagicMock()
    mock_summarizer_agent.create_summary.side_effect = [expected_summary]

    with (
        mock.patch(
            "portia.portia.FinalOutputSummarizer",
            return_value=mock_summarizer_agent,
        ),
        mock.patch.object(
            portia_with_agent_memory,
            "_get_agent_for_step",
            return_value=mock_step_agent,
        ),
    ):
        plan_run = portia_with_agent_memory.run(query)

        # Verify run completed successfully
        assert plan_run.state == PlanRunState.COMPLETE

        # Verify step outputs were stored correctly
        assert plan_run.outputs.step_outputs["$weather"] == AgentMemoryOutput(
            output_name="$weather",
            plan_run_id=plan_run.id,
            summary=weather_summary,
        )
        assert (
            portia_with_agent_memory.storage.get_plan_run_output("$weather", plan_run.id)
            == weather_output
        )
        assert plan_run.outputs.step_outputs["$activities"] == AgentMemoryOutput(
            output_name="$activities",
            plan_run_id=plan_run.id,
            summary=activities_summary,
        )
        assert (
            portia_with_agent_memory.storage.get_plan_run_output("$activities", plan_run.id)
            == activities_output
        )

        # Verify final output and summary
        assert plan_run.outputs.final_output is not None
        assert plan_run.outputs.final_output.get_value() == activities_output.value
        assert plan_run.outputs.final_output.get_summary() == expected_summary


def test_portia_get_final_output_handles_summary_error(portia: Portia) -> None:
    """Test that final output is set even if summary generation fails."""
    (plan, plan_run) = get_test_plan_run()

    # Mock the SummarizerAgent to raise an exception
    mock_agent = mock.MagicMock()
    mock_agent.create_summary.side_effect = Exception("Summary failed")

    with mock.patch(
        "portia.execution_agents.utils.final_output_summarizer.FinalOutputSummarizer",
        return_value=mock_agent,
    ):
        step_output = LocalOutput(value="Some output")
        final_output = portia._get_final_output(plan, plan_run, step_output)  # noqa: SLF001

        # Verify the final output is set without summary
        assert final_output is not None
        assert final_output.get_value() == "Some output"
        assert final_output.get_summary() is None


def test_portia_wait_for_ready_max_retries(portia: Portia) -> None:
    """Test wait for ready with max retries."""
    plan, plan_run = get_test_plan_run()
    plan_run.state = PlanRunState.NEED_CLARIFICATION
    portia.storage.save_plan(plan)
    portia.storage.save_plan_run(plan_run)
    with pytest.raises(InvalidPlanRunStateError):
        portia.wait_for_ready(plan_run, max_retries=0)


def test_portia_wait_for_ready_backoff_period(portia: Portia) -> None:
    """Test wait for ready with backoff period."""
    plan, plan_run = get_test_plan_run()
    plan_run.state = PlanRunState.NEED_CLARIFICATION
    portia.storage.save_plan(plan)
    portia.storage.get_plan_run = mock.MagicMock(return_value=plan_run)
    with pytest.raises(InvalidPlanRunStateError):
        portia.wait_for_ready(plan_run, max_retries=1, backoff_start_time_seconds=0)


def test_portia_resolve_clarification_error(portia: Portia) -> None:
    """Test resolve error."""
    plan, plan_run = get_test_plan_run()
    plan2, plan_run2 = get_test_plan_run()
    clarification = InputClarification(
        user_guidance="",
        argument_name="",
        plan_run_id=plan_run2.id,
    )
    portia.storage.save_plan(plan)
    portia.storage.save_plan_run(plan_run)
    portia.storage.save_plan(plan2)
    portia.storage.save_plan_run(plan_run2)
    with pytest.raises(InvalidPlanRunStateError):
        portia.resolve_clarification(clarification, "test")

    with pytest.raises(InvalidPlanRunStateError):
        portia.resolve_clarification(clarification, "test", plan_run)


def test_portia_resolve_clarification(portia: Portia) -> None:
    """Test resolve success."""
    plan, plan_run = get_test_plan_run()
    clarification = InputClarification(
        user_guidance="",
        argument_name="",
        plan_run_id=plan_run.id,
    )
    plan_run.outputs.clarifications = [clarification]
    portia.storage.save_plan(plan)
    portia.storage.save_plan_run(plan_run)

    plan_run = portia.resolve_clarification(clarification, "test", plan_run)
    assert plan_run.state == PlanRunState.READY_TO_RESUME


def test_portia_get_tool_for_step_none_tool_id() -> None:
    """Test that when step.tool_id is None, LLMTool is used as fallback."""
    portia = Portia(config=get_test_config(), tools=[AdditionTool()])
    plan, plan_run = get_test_plan_run()

    # Create a step with no tool_id
    step = Step(
        task="Some task",
        inputs=[],
        output="$output",
        tool_id=None,
    )

    tool = portia._get_tool_for_step(step, plan_run)  # noqa: SLF001
    assert tool is None


def test_get_llm_tool() -> None:
    """Test special case retrieval of LLMTool as it isn't explicitly in most tool registries."""
    portia = Portia(config=get_test_config(), tools=example_tool_registry)
    plan, plan_run = get_test_plan_run()

    # Create a step with no tool_id
    step = Step(
        task="Some task",
        inputs=[],
        output="$output",
        tool_id=LLMTool.LLM_TOOL_ID,
    )

    tool = portia._get_tool_for_step(step, plan_run)  # noqa: SLF001
    assert tool is not None
    assert isinstance(tool._child_tool, LLMTool)  # noqa: SLF001 # pyright: ignore[reportAttributeAccessIssue]


def test_portia_run_plan(portia: Portia, planning_model: MagicMock) -> None:
    """Test that run_plan calls create_plan_run and resume."""
    query = "example query"

    planning_model.get_structured_response.return_value = StepsOrError(
        steps=[],
        error=None,
    )
    plan = portia.plan(query)

    # Mock the create_plan_run and resume methods
    with (
        mock.patch.object(portia, "create_plan_run") as mockcreate_plan_run,
        mock.patch.object(portia, "resume") as mock_resume,
    ):
        mock_plan_run = MagicMock()
        mock_resumed_plan_run = MagicMock()
        mockcreate_plan_run.return_value = mock_plan_run
        mock_resume.return_value = mock_resumed_plan_run

        result = portia.run_plan(plan)

        mockcreate_plan_run.assert_called_once_with(plan)

        mock_resume.assert_called_once_with(mock_plan_run)

        assert result == mock_resumed_plan_run


def test_portia_handle_clarification(planning_model: MagicMock) -> None:
    """Test that portia can handle a clarification."""
    clarification_handler = TestClarificationHandler()
    portia = Portia(
        config=get_test_config(custom_models={PLANNING_MODEL_KEY: planning_model}),
        tools=[ClarificationTool()],
        execution_hooks=ExecutionHooks(clarification_handler=clarification_handler),
    )
    planning_model.get_structured_response.return_value = StepsOrError(
        steps=[
            Step(
                task="Raise a clarification",
                tool_id="clarification_tool",
                output="$output",
            ),
        ],
        error=None,
    )
    mock_step_agent = mock.MagicMock()
    mock_summarizer_agent = mock.MagicMock()
    mock_summarizer_agent.create_summary.side_effect = "I caught the clarification"
    with (
        mock.patch(
            "portia.portia.FinalOutputSummarizer",
            return_value=mock_summarizer_agent,
        ),
        mock.patch.object(portia, "_get_agent_for_step", return_value=mock_step_agent),
    ):
        plan = portia.plan("Raise a clarification")
        plan_run = portia.create_plan_run(plan)

        mock_step_agent.execute_sync.side_effect = [
            LocalOutput(
                value=InputClarification(
                    plan_run_id=plan_run.id,
                    user_guidance="Handle this clarification",
                    argument_name="raise_clarification",
                ),
            ),
            LocalOutput(value="I caught the clarification"),
        ]
        portia.resume(plan_run)
        assert plan_run.state == PlanRunState.COMPLETE

        # Check that the clarifications were handled correctly
        assert clarification_handler.received_clarification is not None
        assert (
            clarification_handler.received_clarification.user_guidance
            == "Handle this clarification"
        )


def test_portia_error_clarification(portia: Portia, planning_model: MagicMock) -> None:
    """Test that portia can handle an error clarification."""
    planning_model.get_structured_response.return_value = StepsOrError(
        steps=[],
        error=None,
    )
    plan_run = portia.run("test query")

    portia.error_clarification(
        ValueConfirmationClarification(
            plan_run_id=plan_run.id,
            user_guidance="Handle this clarification",
            argument_name="raise_clarification",
        ),
        error=ValueError("test error"),
    )
    assert plan_run.state == PlanRunState.FAILED


def test_portia_error_clarification_with_plan_run(
    portia: Portia,
    planning_model: MagicMock,
) -> None:
    """Test that portia can handle an error clarification."""
    planning_model.get_structured_response.return_value = StepsOrError(
        steps=[],
        error=None,
    )
    plan_run = portia.run("test query")

    portia.error_clarification(
        ValueConfirmationClarification(
            plan_run_id=plan_run.id,
            user_guidance="Handle this clarification",
            argument_name="raise_clarification",
        ),
        error=ValueError("test error"),
        plan_run=plan_run,
    )
    assert plan_run.state == PlanRunState.FAILED


def test_portia_run_with_introspection_skip(portia: Portia, planning_model: MagicMock) -> None:
    """Test run with introspection agent returning SKIP outcome."""
    # Setup mock plan and response
    step1 = Step(task="Step 1", inputs=[], output="$step1_result", condition="some_condition")
    step2 = Step(task="Step 2", inputs=[], output="$step2_result")
    planning_model.get_structured_response.return_value = StepsOrError(
        steps=[step1, step2],
        error=None,
    )

    # Mock introspection agent to return SKIP for first step
    mock_introspection = MagicMock()
    mock_introspection.pre_step_introspection.return_value = PreStepIntrospection(
        outcome=PreStepIntrospectionOutcome.SKIP,
        reason="Condition not met",
    )

    # Mock step agent to return output for second step
    mock_step_agent = MagicMock()
    mock_step_agent.execute_sync.return_value = LocalOutput(value="Step 2 result")

    with (
        mock.patch.object(portia, "_get_introspection_agent", return_value=mock_introspection),
        mock.patch.object(portia, "_get_agent_for_step", return_value=mock_step_agent),
    ):
        plan_run = portia.run("Test query with skipped step")

        # Verify result
        assert plan_run.state == PlanRunState.COMPLETE
        assert "$step1_result" in plan_run.outputs.step_outputs
        assert (
            plan_run.outputs.step_outputs["$step1_result"].get_value()
            == PreStepIntrospectionOutcome.SKIP
        )
        assert "$step2_result" in plan_run.outputs.step_outputs
        assert plan_run.outputs.step_outputs["$step2_result"].get_value() == "Step 2 result"
        assert plan_run.outputs.final_output is not None
        assert plan_run.outputs.final_output.get_value() == "Step 2 result"


def test_portia_run_with_introspection_complete(portia: Portia, planning_model: MagicMock) -> None:
    """Test run with introspection agent returning COMPLETE outcome."""
    # Setup mock plan and response
    step1 = Step(task="Step 1", inputs=[], output="$step1_result")
    step2 = Step(task="Step 2", inputs=[], output="$step2_result", condition="some_condition")
    step3 = Step(task="Step 3", inputs=[], output="$step3_result")
    planning_model.get_structured_response.return_value = StepsOrError(
        steps=[step1, step2, step3],
        error=None,
    )

    # Mock step agent for first step
    mock_step_agent = MagicMock()
    mock_step_agent.execute_sync.return_value = LocalOutput(value="Step 1 result")

    # Configure the COMPLETE outcome for the introspection agent
    mock_introspection_complete = PreStepIntrospection(
        outcome=PreStepIntrospectionOutcome.COMPLETE,
        reason="Remaining steps cannot be executed",
    )

    def custom_handle_introspection(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202, ARG001
        plan_run: PlanRun = kwargs.get("plan_run")  # type: ignore  # noqa: PGH003

        if plan_run.current_step_index == 1:
            plan_run.outputs.step_outputs["$step2_result"] = LocalOutput(
                value=PreStepIntrospectionOutcome.COMPLETE,
                summary="Remaining steps cannot be executed",
            )
            plan_run.outputs.final_output = LocalOutput(
                value="Step 1 result",
                summary="Execution completed early",
            )
            plan_run.state = PlanRunState.COMPLETE

            return (plan_run, mock_introspection_complete)

        # Otherwise continue normally
        return (
            plan_run,
            PreStepIntrospection(
                outcome=PreStepIntrospectionOutcome.CONTINUE,
                reason="Condition met",
            ),
        )

    with (
        mock.patch.object(portia, "_handle_introspection_outcome", custom_handle_introspection),
        mock.patch.object(portia, "_get_agent_for_step", return_value=mock_step_agent),
    ):
        # Run the test
        plan_run = portia.run("Test query with early completed execution")

        # Verify result based on our simulated outcomes
        assert plan_run.state == PlanRunState.COMPLETE
        assert "$step2_result" in plan_run.outputs.step_outputs
        assert (
            plan_run.outputs.step_outputs["$step2_result"].get_value()
            == PreStepIntrospectionOutcome.COMPLETE
        )
        assert plan_run.outputs.final_output is not None
        assert plan_run.outputs.final_output.get_summary() == "Execution completed early"


def test_portia_run_with_introspection_fail(portia: Portia, planning_model: MagicMock) -> None:
    """Test run with introspection agent returning FAIL outcome."""
    # Setup mock plan and response
    step1 = Step(task="Step 1", inputs=[], output="$step1_result")
    step2 = Step(task="Step 2", inputs=[], output="$step2_result", condition="some_condition")
    planning_model.get_structured_response.return_value = StepsOrError(
        steps=[step1, step2],
        error=None,
    )

    # Mock step agent for first step
    mock_step_agent = MagicMock()
    mock_step_agent.execute_sync.return_value = LocalOutput(value="Step 1 result")

    # Configure the FAIL outcome
    mock_introspection_fail = PreStepIntrospection(
        outcome=PreStepIntrospectionOutcome.FAIL,
        reason="Missing required data",
    )

    def custom_handle_introspection(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202, ARG001
        plan_run: PlanRun = kwargs.get("plan_run")  # type: ignore  # noqa: PGH003
        # If this is step 1, simulate a FAIL outcome
        if plan_run.current_step_index == 1:
            # Modify the plan_run to look like it failed
            failed_output = LocalOutput(
                value=PreStepIntrospectionOutcome.FAIL,
                summary="Missing required data",
            )
            plan_run.outputs.step_outputs["$step2_result"] = failed_output
            plan_run.outputs.final_output = failed_output
            plan_run.state = PlanRunState.FAILED

            # Return FAIL outcome
            return (plan_run, mock_introspection_fail)

        # Otherwise continue normally
        return (
            plan_run,
            PreStepIntrospection(
                outcome=PreStepIntrospectionOutcome.CONTINUE,
                reason="Condition met",
            ),
        )

    with (
        mock.patch.object(portia, "_handle_introspection_outcome", custom_handle_introspection),
        mock.patch.object(portia, "_get_agent_for_step", return_value=mock_step_agent),
    ):
        # Run the test
        plan_run = portia.run("Test query with failed execution")

        # Verify the expected outcome
        assert plan_run.state == PlanRunState.FAILED
        assert "$step2_result" in plan_run.outputs.step_outputs
        assert (
            plan_run.outputs.step_outputs["$step2_result"].get_value()
            == PreStepIntrospectionOutcome.FAIL
        )
        assert plan_run.outputs.final_output is not None
        assert plan_run.outputs.final_output.get_value() == PreStepIntrospectionOutcome.FAIL
        assert plan_run.outputs.final_output.get_summary() == "Missing required data"


def test_handle_introspection_outcome_complete(portia: Portia) -> None:
    """Test the actual implementation of _handle_introspection_outcome for COMPLETE outcome."""
    # Create a plan with conditions
    step = Step(task="Test step", inputs=[], output="$test_output", condition="some_condition")
    plan = Plan(
        plan_context=PlanContext(query="test query", tool_ids=[]),
        steps=[step],
    )
    plan_run = PlanRun(
        plan_id=plan.id,
        current_step_index=0,
        state=PlanRunState.IN_PROGRESS,
    )

    mock_introspection = MagicMock()
    mock_introspection.pre_step_introspection.return_value = PreStepIntrospection(
        outcome=PreStepIntrospectionOutcome.COMPLETE,
        reason="Stopping execution",
    )

    # Mock the _get_final_output method to return a predefined output
    mock_final_output = LocalOutput(value="Final result", summary="Final summary")
    with mock.patch.object(portia, "_get_final_output", return_value=mock_final_output):
        # Call the actual method (not mocked)
        previous_output = LocalOutput(value="Previous step result")
        updated_plan_run, outcome = portia._handle_introspection_outcome(  # noqa: SLF001
            introspection_agent=mock_introspection,
            plan=plan,
            plan_run=plan_run,
            last_executed_step_output=previous_output,
        )

        # Verify the outcome
        assert outcome.outcome == PreStepIntrospectionOutcome.COMPLETE
        assert outcome.reason == "Stopping execution"

        # Verify plan_run was updated correctly
        assert (
            updated_plan_run.outputs.step_outputs["$test_output"].get_value()
            == PreStepIntrospectionOutcome.COMPLETE
        )
        assert (
            updated_plan_run.outputs.step_outputs["$test_output"].get_summary()
            == "Stopping execution"
        )
        assert updated_plan_run.outputs.final_output == mock_final_output
        assert updated_plan_run.state == PlanRunState.COMPLETE


def test_handle_introspection_outcome_fail(portia: Portia) -> None:
    """Test the actual implementation of _handle_introspection_outcome for FAIL outcome."""
    # Create a plan with conditions
    step = Step(task="Test step", inputs=[], output="$test_output", condition="some_condition")
    plan = Plan(
        plan_context=PlanContext(query="test query", tool_ids=[]),
        steps=[step],
    )
    plan_run = PlanRun(
        plan_id=plan.id,
        current_step_index=0,
        state=PlanRunState.IN_PROGRESS,
    )

    # Mock the introspection agent to return FAIL
    mock_introspection = MagicMock()
    mock_introspection.pre_step_introspection.return_value = PreStepIntrospection(
        outcome=PreStepIntrospectionOutcome.FAIL,
        reason="Execution failed",
    )

    # Call the actual method (not mocked)
    previous_output = LocalOutput(value="Previous step result")
    updated_plan_run, outcome = portia._handle_introspection_outcome(  # noqa: SLF001
        introspection_agent=mock_introspection,
        plan=plan,
        plan_run=plan_run,
        last_executed_step_output=previous_output,
    )

    # Verify the outcome
    assert outcome.outcome == PreStepIntrospectionOutcome.FAIL
    assert outcome.reason == "Execution failed"

    # Verify plan_run was updated correctly
    assert (
        updated_plan_run.outputs.step_outputs["$test_output"].get_value()
        == PreStepIntrospectionOutcome.FAIL
    )
    assert updated_plan_run.outputs.step_outputs["$test_output"].get_summary() == "Execution failed"
    assert updated_plan_run.outputs.final_output is not None
    assert updated_plan_run.outputs.final_output.get_value() == PreStepIntrospectionOutcome.FAIL
    assert updated_plan_run.outputs.final_output.get_summary() == "Execution failed"
    assert updated_plan_run.state == PlanRunState.FAILED


def test_handle_introspection_outcome_skip(portia: Portia) -> None:
    """Test the actual implementation of _handle_introspection_outcome for SKIP outcome."""
    # Create a plan with conditions
    step = Step(task="Test step", inputs=[], output="$test_output", condition="some_condition")
    plan = Plan(
        plan_context=PlanContext(query="test query", tool_ids=[]),
        steps=[step],
    )
    plan_run = PlanRun(
        plan_id=plan.id,
        current_step_index=0,
        state=PlanRunState.IN_PROGRESS,
    )

    # Mock the introspection agent to return SKIP
    mock_introspection = MagicMock()
    mock_introspection.pre_step_introspection.return_value = PreStepIntrospection(
        outcome=PreStepIntrospectionOutcome.SKIP,
        reason="Skipping step",
    )

    # Call the actual method (not mocked)
    previous_output = LocalOutput(value="Previous step result")
    updated_plan_run, outcome = portia._handle_introspection_outcome(  # noqa: SLF001
        introspection_agent=mock_introspection,
        plan=plan,
        plan_run=plan_run,
        last_executed_step_output=previous_output,
    )

    # Verify the outcome
    assert outcome.outcome == PreStepIntrospectionOutcome.SKIP
    assert outcome.reason == "Skipping step"

    # Verify plan_run was updated correctly
    assert (
        updated_plan_run.outputs.step_outputs["$test_output"].get_value()
        == PreStepIntrospectionOutcome.SKIP
    )
    assert updated_plan_run.outputs.step_outputs["$test_output"].get_summary() == "Skipping step"
    assert updated_plan_run.state == PlanRunState.IN_PROGRESS  # State should remain IN_PROGRESS


def test_handle_introspection_outcome_no_condition(portia: Portia) -> None:
    """Test _handle_introspection_outcome when step has no condition."""
    # Create a plan with a step that has no condition
    step = Step(task="Test step", inputs=[], output="$test_output")  # No condition
    plan = Plan(
        plan_context=PlanContext(query="test query", tool_ids=[]),
        steps=[step],
    )
    plan_run = PlanRun(
        plan_id=plan.id,
        current_step_index=0,
        state=PlanRunState.IN_PROGRESS,
    )

    # Mock the introspection agent (should not be called)
    mock_introspection = MagicMock()

    # Call the actual method
    previous_output = LocalOutput(value="Previous step result")
    updated_plan_run, outcome = portia._handle_introspection_outcome(  # noqa: SLF001
        introspection_agent=mock_introspection,
        plan=plan,
        plan_run=plan_run,
        last_executed_step_output=previous_output,
    )

    # Verify default outcome is CONTINUE
    assert outcome.outcome == PreStepIntrospectionOutcome.CONTINUE
    assert outcome.reason == "No condition to evaluate."

    # The introspection agent should not be called
    mock_introspection.pre_step_introspection.assert_not_called()

    # Plan run should be unchanged (no step outputs added)
    assert "$test_output" not in updated_plan_run.outputs.step_outputs
    assert updated_plan_run.state == PlanRunState.IN_PROGRESS


def test_portia_resume_with_skipped_steps(portia: Portia) -> None:
    """Test resuming a plan run with skipped steps and verifying final output.

    This test verifies:
    1. Resuming from a middle index works correctly
    2. Steps marked as SKIPPED are properly skipped during execution
    3. The final output is correctly computed from the last non-SKIPPED step
    """
    # Create a plan with multiple steps
    step1 = Step(task="Step 1", inputs=[], output="$step1_result")
    step2 = Step(task="Step 2", inputs=[], output="$step2_result", condition="true")
    step3 = Step(task="Step 3", inputs=[], output="$step3_result", condition="false")
    step4 = Step(task="Step 4", inputs=[], output="$step4_result", condition="false")
    plan = Plan(
        plan_context=PlanContext(query="Test query with skips", tool_ids=[]),
        steps=[step1, step2, step3, step4],
    )

    # Create a plan run that's partially completed (step1 is done)
    plan_run = PlanRun(
        plan_id=plan.id,
        current_step_index=1,  # Resume from step 2
        state=PlanRunState.IN_PROGRESS,
        outputs=PlanRunOutputs(
            step_outputs={
                "$step1_result": LocalOutput(value="Step 1 result", summary="Summary of step 1"),
            },
        ),
    )

    # Mock the storage to return our plan
    portia.storage.save_plan(plan)
    portia.storage.save_plan_run(plan_run)

    # Mock introspection agent to SKIP steps 3 and 4
    mock_introspection = MagicMock()

    def mock_introspection_outcome(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202, ARG001
        plan_run = kwargs.get("plan_run")
        if plan_run.current_step_index in (2, 3):  # pyright: ignore[reportOptionalMemberAccess] # Skip both step3 and step4
            return PreStepIntrospection(
                outcome=PreStepIntrospectionOutcome.SKIP,
                reason="Condition is false",
            )
        return PreStepIntrospection(
            outcome=PreStepIntrospectionOutcome.CONTINUE,
            reason="Continue execution",
        )

    mock_introspection.pre_step_introspection.side_effect = mock_introspection_outcome

    # Mock step agent to return expected output for step 2 only (steps 3 and 4 will be skipped)
    mock_step_agent = MagicMock()
    mock_step_agent.execute_sync.return_value = LocalOutput(
        value="Step 2 result",
        summary="Summary of step 2",
    )

    # Mock the final output summarizer
    expected_summary = "Combined summary of steps 1 and 2"
    mock_summarizer = MagicMock()
    mock_summarizer.create_summary.return_value = expected_summary

    with (
        mock.patch.object(portia, "_get_introspection_agent", return_value=mock_introspection),
        mock.patch.object(portia, "_get_agent_for_step", return_value=mock_step_agent),
        mock.patch("portia.portia.FinalOutputSummarizer", return_value=mock_summarizer),
    ):
        result_plan_run = portia.resume(plan_run)

        assert result_plan_run.state == PlanRunState.COMPLETE

        assert result_plan_run.outputs.step_outputs["$step1_result"].get_value() == "Step 1 result"
        assert result_plan_run.outputs.step_outputs["$step2_result"].get_value() == "Step 2 result"
        assert (
            result_plan_run.outputs.step_outputs["$step3_result"].get_value()
            == PreStepIntrospectionOutcome.SKIP
        )
        assert (
            result_plan_run.outputs.step_outputs["$step4_result"].get_value()
            == PreStepIntrospectionOutcome.SKIP
        )
        assert result_plan_run.outputs.final_output is not None
        assert result_plan_run.outputs.final_output.get_value() == "Step 2 result"
        assert result_plan_run.outputs.final_output.get_summary() == expected_summary
        assert result_plan_run.current_step_index == 3
