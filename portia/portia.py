"""Portia classes that plan and execute runs for queries.

This module contains the core classes responsible for generating, managing, and executing plans
in response to queries. The `Portia` class serves as the main entry point, orchestrating the
planning and execution process. It uses various agents and tools to carry out tasks step by step,
saving the state of the run at each stage. It also handles error cases, clarification
requests, and run state transitions.

The `Portia` class provides methods to:

- Generate a plan for executing a query.
- Create and manage runs.
- Execute runs step by step, using agents to handle the execution of tasks.
- Resolve clarifications required during the execution of runs.
- Wait for runs to reach a state where they can be resumed.

Modules in this file work with different storage backends (memory, disk, cloud) and can handle
complex queries using various planning and execution agent configurations.

"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from portia.clarification import (
    Clarification,
    ClarificationCategory,
)
from portia.config import Config, ExecutionAgentType, PlanningAgentType, StorageClass
from portia.errors import (
    InvalidPlanRunStateError,
    PlanError,
)
from portia.execution_agents.base_execution_agent import Output
from portia.execution_agents.default_execution_agent import DefaultExecutionAgent
from portia.execution_agents.one_shot_agent import OneShotAgent
from portia.execution_agents.utils.final_output_summarizer import FinalOutputSummarizer
from portia.execution_context import (
    execution_context,
    get_execution_context,
    is_execution_context_set,
)
from portia.logger import logger, logger_manager
from portia.open_source_tools.llm_tool import LLMTool
from portia.plan import Plan, PlanContext, ReadOnlyPlan, ReadOnlyStep, Step
from portia.plan_run import PlanRun, PlanRunState, PlanRunUUID, ReadOnlyPlanRun
from portia.planning_agents.default_planning_agent import DefaultPlanningAgent
from portia.storage import (
    DiskFileStorage,
    InMemoryStorage,
    PortiaCloudStorage,
)
from portia.tool import ToolRunContext
from portia.tool_registry import DefaultToolRegistry, InMemoryToolRegistry, ToolRegistry
from portia.tool_wrapper import ToolCallWrapper

if TYPE_CHECKING:
    from portia.execution_agents.base_execution_agent import BaseExecutionAgent
    from portia.planning_agents.base_planning_agent import BasePlanningAgent
    from portia.tool import Tool


class Portia:
    """Portia client is the top level abstraction and entrypoint for most programs using the SDK.

    It is responsible for intermediating planning via PlanningAgents and
    execution via ExecutionAgents.
    """

    def __init__(
        self,
        config: Config | None = None,
        tools: ToolRegistry | list[Tool] | None = None,
    ) -> None:
        """Initialize storage and tools.

        Args:
            config (Config): The configuration to initialize the Portia client. If not provided, the
                default configuration will be used.
            tools (ToolRegistry | list[Tool]): The registry or list of tools to use. If not
                provided, the open source tool registry will be used, alongside the default tools
                from Portia cloud if a Portia API key is set.

        """
        self.config = config if config else Config.from_default()
        logger_manager.configure_from_config(self.config)

        if isinstance(tools, ToolRegistry):
            self.tool_registry = tools
        elif isinstance(tools, list):
            self.tool_registry = InMemoryToolRegistry.from_local_tools(tools)
        else:
            self.tool_registry = DefaultToolRegistry(self.config)

        match self.config.storage_class:
            case StorageClass.MEMORY:
                self.storage = InMemoryStorage()
            case StorageClass.DISK:
                self.storage = DiskFileStorage(storage_dir=self.config.must_get("storage_dir", str))
            case StorageClass.CLOUD:
                self.storage = PortiaCloudStorage(config=self.config)

    def run(
        self,
        query: str,
        tools: list[Tool] | list[str] | None = None,
        example_plans: list[Plan] | None = None,
    ) -> PlanRun:
        """End-to-end function to generate a plan and then execute it.

        This is the simplest way to plan and execute a query using the SDK.

        Args:
            query (str): The query to be executed.
            tools (list[Tool] | list[str] | None): List of tools to use for the query.
            If not provided all tools in the registry will be used.
            example_plans (list[Plan] | None): Optional list of example plans. If not
            provide a default set of example plans will be used.

        Returns:
            PlanRun: The run resulting from executing the query.

        """
        plan = self.plan(query, tools, example_plans)
        plan_run = self._create_plan_run(plan)
        return self.resume(plan_run)

    def plan(
        self,
        query: str,
        tools: list[Tool] | list[str] | None = None,
        example_plans: list[Plan] | None = None,
    ) -> Plan:
        """Plans how to do the query given the set of tools and any examples.

        Args:
            query (str): The query to generate the plan for.
            tools (list[Tool] | list[str] | None): List of tools to use for the query.
            If not provided all tools in the registry will be used.
            example_plans (list[Plan] | None): Optional list of example plans. If not
            provide a default set of example plans will be used.

        Returns:
            Plan: The plan for executing the query.

        Raises:
            PlanError: If there is an error while generating the plan.

        """
        if isinstance(tools, list):
            tools = [
                self.tool_registry.get_tool(tool) if isinstance(tool, str) else tool
                for tool in tools
            ]

        if not tools:
            tools = self.tool_registry.match_tools(query)

        logger().debug(f"Running planning_agent for query - {query}")
        planning_agent = self._get_planning_agent()
        outcome = planning_agent.generate_steps_or_error(
            ctx=get_execution_context(),
            query=query,
            tool_list=tools,
            examples=example_plans,
        )
        if outcome.error:
            logger().error(f"Error in planning - {outcome.error}")
            raise PlanError(outcome.error)
        plan = Plan(
            plan_context=PlanContext(
                query=query,
                tool_ids=[tool.id for tool in tools],
            ),
            steps=outcome.steps,
        )
        self.storage.save_plan(plan)
        logger().info(
            f"Plan created with {len(plan.steps)} steps",
            extra={"plan": plan.id},
        )
        logger().debug(
            "Plan: {plan}",
            extra={"plan": plan.id},
            plan=plan.model_dump_json(indent=4),
        )

        return plan

    def run_plan(self, plan: Plan) -> PlanRun:
        """Run a plan.

        Args:
            plan (Plan): The plan to run.

        Returns:
            PlanRun: The resulting PlanRun object.

        """
        plan_run = self._create_plan_run(plan)
        return self.resume(plan_run)

    def resume(
        self,
        plan_run: PlanRun | None = None,
        plan_run_id: PlanRunUUID | str | None = None,
    ) -> PlanRun:
        """Resume a PlanRun.

        Args:
            plan_run (PlanRun | None): The PlanRun to resume. Defaults to None.
            plan_run_id (RunUUID | str | None): The ID of the PlanRun to resume. Defaults to
                None.

        Returns:
            PlanRun: The resulting PlanRun after execution.

        Raises:
            ValueError: If neither plan_run nor plan_run_id is provided.
            InvalidPlanRunStateError: If the plan run is not in a valid state to be resumed.

        """
        if not plan_run:
            if not plan_run_id:
                raise ValueError("Either plan_run or plan_run_id must be provided")

            parsed_id = (
                PlanRunUUID.from_string(plan_run_id)
                if isinstance(plan_run_id, str)
                else plan_run_id
            )
            plan_run = self.storage.get_plan_run(parsed_id)

        if plan_run.state not in [
            PlanRunState.NOT_STARTED,
            PlanRunState.IN_PROGRESS,
            PlanRunState.NEED_CLARIFICATION,
            PlanRunState.READY_TO_RESUME,
        ]:
            raise InvalidPlanRunStateError(plan_run.id)

        plan = self.storage.get_plan(plan_id=plan_run.plan_id)

        # if the run has execution context associated, but none is set then use it
        if not is_execution_context_set():
            with execution_context(plan_run.execution_context):
                return self._execute_plan_run(plan, plan_run)

        # if there is execution context set, make sure we update the run before running
        plan_run.execution_context = get_execution_context()
        return self._execute_plan_run(plan, plan_run)

    def resolve_clarification(
        self,
        clarification: Clarification,
        response: object,
        plan_run: PlanRun | None = None,
    ) -> PlanRun:
        """Resolve a clarification updating the run state as needed.

        Args:
            clarification (Clarification): The clarification to resolve.
            response (object): The response to the clarification.
            plan_run (PlanRun | None): Optional - the plan run being updated.

        Returns:
            PlanRun: The updated PlanRun.

        """
        if plan_run is None:
            plan_run = self.storage.get_plan_run(clarification.plan_run_id)

        matched_clarification = next(
            (c for c in plan_run.outputs.clarifications if c.id == clarification.id),
            None,
        )

        if not matched_clarification:
            raise InvalidPlanRunStateError("Could not match clarification to run")

        matched_clarification.resolved = True
        matched_clarification.response = response

        if len(plan_run.get_outstanding_clarifications()) == 0:
            plan_run.state = PlanRunState.READY_TO_RESUME

        self.storage.save_plan_run(plan_run)
        return plan_run

    def wait_for_ready(
        self,
        plan_run: PlanRun,
        max_retries: int = 6,
        backoff_start_time_seconds: int = 7 * 60,
        backoff_time_seconds: int = 2,
    ) -> PlanRun:
        """Wait for the run to be in a state that it can be re-plan_run.

        This is generally because there are outstanding clarifications that need to be resolved.

        Args:
            plan_run (PlanRun): The PlanRun to wait for.
            max_retries (int): The maximum number of retries to wait for the run to be ready
                after the backoff period starts.
            backoff_start_time_seconds (int): The time after which the backoff period starts.
            backoff_time_seconds (int): The time to wait between retries after the backoff period
                starts.

        Returns:
            PlanRun: The updated PlanRun once it is ready to be re-plan_run.

        Raises:
            InvalidRunStateError: If the run cannot be waited for.

        """
        start_time = time.time()
        tries = 0
        if plan_run.state not in [
            PlanRunState.IN_PROGRESS,
            PlanRunState.NOT_STARTED,
            PlanRunState.READY_TO_RESUME,
            PlanRunState.NEED_CLARIFICATION,
        ]:
            raise InvalidPlanRunStateError("Cannot wait for run that is not ready to run")

        # These states can continue straight away
        if plan_run.state in [
            PlanRunState.IN_PROGRESS,
            PlanRunState.NOT_STARTED,
            PlanRunState.READY_TO_RESUME,
        ]:
            return plan_run

        plan = self.storage.get_plan(plan_run.plan_id)
        plan_run = self.storage.get_plan_run(plan_run.id)
        current_step_clarifications = plan_run.get_clarifications_for_step()
        while plan_run.state != PlanRunState.READY_TO_RESUME:
            if tries >= max_retries:
                raise InvalidPlanRunStateError("Run is not ready to resume after max retries")

            # if we've waited longer than the backoff time, start the backoff period
            if time.time() - start_time > backoff_start_time_seconds:
                tries += 1
                backoff_time_seconds *= 2

            # wait a couple of seconds as we're long polling
            time.sleep(backoff_time_seconds)

            step = plan.steps[plan_run.current_step_index]
            next_tool = self._get_tool_for_step(step, plan_run)
            if next_tool:
                tool_ready = next_tool.ready(
                    ToolRunContext(
                        execution_context=plan_run.execution_context,
                        plan_run_id=plan_run.id,
                        config=self.config,
                        clarifications=current_step_clarifications,
                    ),
                )
                logger().debug(f"Tool state for {next_tool.name} is ready={tool_ready}")
                if tool_ready:
                    for clarification in current_step_clarifications:
                        if clarification.category is ClarificationCategory.ACTION:
                            clarification.resolved = True
                            clarification.response = "complete"
                    plan_run.state = PlanRunState.READY_TO_RESUME
                    self.storage.save_plan_run(plan_run)

            logger().debug(f"New run state for {plan_run.id} is {plan_run.state}")

        logger().info(f"Run {plan_run.id} is ready to resume")

        return plan_run

    def _create_plan_run(self, plan: Plan) -> PlanRun:
        """Create a PlanRun from a Plan.

        Args:
            plan (Plan): The plan to create a plan run from.

        Returns:
            PlanRun: The created PlanRun object.

        """
        plan_run = PlanRun(
            plan_id=plan.id,
            state=PlanRunState.NOT_STARTED,
            execution_context=get_execution_context(),
        )
        self.storage.save_plan_run(plan_run)
        return plan_run

    def _execute_plan_run(self, plan: Plan, plan_run: PlanRun) -> PlanRun:
        """Execute the run steps, updating the run state as needed.

        Args:
            plan (Plan): The plan to execute.
            plan_run (PlanRun): The plan run to execute.

        Returns:
            Run: The updated run after execution.

        """
        plan_run.state = PlanRunState.IN_PROGRESS
        self.storage.save_plan_run(plan_run)
        logger().debug(
            f"Executing run from step {plan_run.current_step_index}",
            extra={"plan": plan.id, "plan_run": plan_run.id},
        )
        for index in range(plan_run.current_step_index, len(plan.steps)):
            step = plan.steps[index]
            plan_run.current_step_index = index
            logger().debug(
                f"Executing step {index}: {step.task}",
                extra={"plan": plan.id, "plan_run": plan_run.id},
            )
            # we pass read only copies of the state to the agent so that the portia remains
            # responsible for handling the output of the agent and updating the state.
            agent = self._get_agent_for_step(
                step=ReadOnlyStep.from_step(step),
                plan_run=ReadOnlyPlanRun.from_plan_run(plan_run),
            )
            logger().debug(
                f"Using agent: {type(agent)}",
                extra={"plan": plan.id, "plan_run": plan_run.id},
            )
            try:
                step_output = agent.execute_sync()
            except Exception as e:  # noqa: BLE001 - We want to capture all failures here
                error_output = Output(value=str(e))
                plan_run.outputs.step_outputs[step.output] = error_output
                plan_run.state = PlanRunState.FAILED
                plan_run.outputs.final_output = error_output
                self.storage.save_plan_run(plan_run)
                logger().error(
                    "error: {error}",
                    error=e,
                    extra={"plan": plan.id, "plan_run": plan_run.id},
                )
                logger().debug(
                    f"Final run status: {plan_run.state}",
                    extra={"plan": plan.id, "plan_run": plan_run.id},
                )
                return plan_run
            else:
                plan_run.outputs.step_outputs[step.output] = step_output
                logger().debug(
                    "Step output - {output}",
                    extra={"plan": plan.id, "plan_run": plan_run.id},
                    output=str(step_output.value),
                )

            if self._handle_clarifications(plan_run, step_output, plan):
                return plan_run

            # set final output if is last step (accounting for zero index)
            if index == len(plan.steps) - 1:
                plan_run.outputs.final_output = self._get_final_output(plan, plan_run, step_output)

            # persist at the end of each step
            self.storage.save_plan_run(plan_run)
            logger().debug(
                "New PlanRun State: {plan_run}",
                extra={"plan": plan.id, "plan_run": plan_run.id},
                plan_run=plan_run.model_dump_json(indent=4),
            )

        plan_run.state = PlanRunState.COMPLETE
        self.storage.save_plan_run(plan_run)
        logger().debug(
            f"Final run status: {plan_run.state}",
            extra={"plan": plan.id, "plan_run": plan_run.id},
        )
        if plan_run.outputs.final_output:
            logger().info(
                "{output}",
                extra={"plan": plan.id, "plan_run": plan_run.id},
                output=str(plan_run.outputs.final_output.value),
            )
        return plan_run

    def _get_planning_agent(self) -> BasePlanningAgent:
        """Get the planning_agent based on the configuration.

        Returns:
            BasePlanningAgent: The planning agent to be used for generating plans.

        """
        cls: type[BasePlanningAgent]
        match self.config.planning_agent_type:
            case PlanningAgentType.DEFAULT:
                cls = DefaultPlanningAgent

        return cls(self.config)

    def _get_final_output(self, plan: Plan, plan_run: PlanRun, step_output: Output) -> Output:
        """Get the final output and add summarization to it.

        Args:
            plan (Plan): The plan to execute.
            plan_run (PlanRun): The PlanRun to execute.
            step_output (Output): The output of the last step.

        """
        final_output = Output(
            value=step_output.value,
            summary=None,
        )

        try:
            summarizer = FinalOutputSummarizer(config=self.config)
            summary = summarizer.create_summary(
                plan_run=ReadOnlyPlanRun.from_plan_run(plan_run),
                plan=ReadOnlyPlan.from_plan(plan),
            )
            final_output.summary = summary

        except Exception as e:  # noqa: BLE001
            logger().warning(f"Error summarising run: {e}")

        return final_output

    def _handle_clarifications(self, plan_run: PlanRun, step_output: Output, plan: Plan) -> bool:
        """Handle any clarifications needed during run execution.

        Args:
            plan_run (PlanRun): The PlanRun to execute.
            step_output (Output): The output of the last step.
            plan (Plan): The plan to execute.

        Returns:
            bool: True if clarification is needed and run execution should stop.

        """
        if isinstance(step_output.value, Clarification) or (
            isinstance(step_output.value, list)
            and len(step_output.value) > 0
            and all(isinstance(item, Clarification) for item in step_output.value)
        ):
            new_clarifications = (
                [step_output.value]
                if isinstance(step_output.value, Clarification)
                else step_output.value
            )
            for clarification in new_clarifications:
                clarification.step = plan_run.current_step_index

            plan_run.outputs.clarifications = plan_run.outputs.clarifications + new_clarifications
            plan_run.state = PlanRunState.NEED_CLARIFICATION
            self.storage.save_plan_run(plan_run)
            logger().info(
                f"{len(new_clarifications)} Clarification(s) requested",
                extra={"plan": plan.id, "plan_run": plan_run.id},
            )
            return True
        return False

    def _get_tool_for_step(self, step: Step, plan_run: PlanRun) -> Tool | None:
        if not step.tool_id:
            return None
        if step.tool_id == LLMTool.LLM_TOOL_ID:
            # Special case LLMTool so it doesn't need to be in all tool registries
            child_tool = LLMTool()
        else:
            child_tool = self.tool_registry.get_tool(step.tool_id)
        return ToolCallWrapper(
            child_tool=child_tool,
            storage=self.storage,
            plan_run=plan_run,
        )

    def _get_agent_for_step(
        self,
        step: Step,
        plan_run: PlanRun,
    ) -> BaseExecutionAgent:
        """Get the appropriate agent for executing a given step.

        Args:
            step (Step): The step for which the agent is needed.
            plan_run (PlanRun): The run associated with the step.

        Returns:
            BaseAgent: The agent to execute the step.

        """
        tool = self._get_tool_for_step(step, plan_run)
        cls: type[BaseExecutionAgent]
        match self.config.execution_agent_type:
            case ExecutionAgentType.ONE_SHOT:
                cls = OneShotAgent
            case ExecutionAgentType.DEFAULT:
                cls = DefaultExecutionAgent
        return cls(
            step,
            plan_run,
            self.config,
            tool,
        )

