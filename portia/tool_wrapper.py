"""Tool Wrapper that intercepts run calls and records them.

This module contains the `ToolCallWrapper` class, which wraps around an existing tool and records
information about the tool's execution, such as input, output, latency, and status. The recorded
data is stored in `AdditionalStorage` for later use.

Classes:
    ToolCallWrapper: A wrapper that intercepts tool calls, records execution data, and stores it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict

from portia.clarification import Clarification
from portia.common import combine_args_kwargs
from portia.execution_agents.output import LocalOutput
from portia.logger import logger
from portia.storage import AdditionalStorage, ToolCallRecord, ToolCallStatus
from portia.tool import Tool, ToolRunContext

if TYPE_CHECKING:
    from portia.plan_run import PlanRun


class ToolCallWrapper(Tool):
    """Tool Wrapper that records calls to its child tool and sends them to the AdditionalStorage.

    This class is a wrapper around a child tool. It captures the input and output, measures latency,
    and records the status of the execution. The results are then stored in the provided
    `AdditionalStorage`.

    Attributes:
        model_config (ConfigDict): Pydantic configuration that allows arbitrary types.
        _child_tool (Tool): The child tool to be wrapped and executed.
        _storage (AdditionalStorage): Storage mechanism to save tool call records.
        _plan_run (PlanRun): The run context for the current execution.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _child_tool: Tool
    _storage: AdditionalStorage
    _plan_run: PlanRun

    def __init__(self, child_tool: Tool, storage: AdditionalStorage, plan_run: PlanRun) -> None:
        """Initialize parent fields using child_tool's attributes.

        Args:
            child_tool (Tool): The child tool to be wrapped.
            storage (AdditionalStorage): The storage to save execution records.
            plan_run (PlanRun): The PlanRun to execute.

        """
        super().__init__(
            id=child_tool.id,
            name=child_tool.name,
            description=child_tool.description,
            args_schema=child_tool.args_schema,
            output_schema=child_tool.output_schema,
            should_summarize=child_tool.should_summarize,
        )
        self._child_tool = child_tool
        self._storage = storage
        self._plan_run = plan_run

    def ready(self, ctx: ToolRunContext) -> bool:
        """Check if the child tool is ready.

        Args:
            ctx (ToolRunContext): Context of the tool run

        Returns:
            bool: Whether the tool is ready to run

        """
        return self._child_tool.ready(ctx)

    def run(self, ctx: ToolRunContext, *args: Any, **kwargs: Any) -> Any | Clarification:  # noqa: ANN401
        """Run the child tool and store the outcome.

        This method executes the child tool with the provided arguments, records the input,
        output, latency, and status of the execution, and stores the details in `AdditionalStorage`.

        Args:
            ctx (ToolRunContext): The context containing user data and metadata.
            *args (Any): Positional arguments for the child tool.
            **kwargs (Any): Keyword arguments for the child tool.

        Returns:
            Any | Clarification: The output of the child tool or a clarification request.

        Raises:
            Exception: If an error occurs during execution, the exception is logged, and the
                status is set to `FAILED`.

        """
        # initialize empty call record

        record = ToolCallRecord(
            input=combine_args_kwargs(*args, **kwargs),
            output=None,
            latency_seconds=0,
            tool_name=self._child_tool.name,
            plan_run_id=self._plan_run.id,
            step=self._plan_run.current_step_index,
            end_user_id=ctx.execution_context.end_user_id,
            additional_data=ctx.execution_context.additional_data,
            status=ToolCallStatus.IN_PROGRESS,
        )
        logger().info(
            f"Invoking {record.tool_name} with args: {record.input}",
        )
        start_time = datetime.now(tz=UTC)
        try:
            output = self._child_tool.run(ctx, *args, **kwargs)
        except Exception as e:
            record.output = str(e)
            record.latency_seconds = (datetime.now(tz=UTC) - start_time).total_seconds()
            record.status = ToolCallStatus.FAILED
            self._storage.save_tool_call(record)
            raise
        else:
            if isinstance(output, Clarification):
                record.status = ToolCallStatus.NEED_CLARIFICATION
                record.output = output.model_dump(mode="json")
            elif output is None:
                record.output = LocalOutput(value=output).model_dump(mode="json")
                record.status = ToolCallStatus.SUCCESS
            else:
                record.output = output
                record.status = ToolCallStatus.SUCCESS
            record.latency_seconds = (datetime.now(tz=UTC) - start_time).total_seconds()
            self._storage.save_tool_call(record)
        return output
