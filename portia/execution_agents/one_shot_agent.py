"""A simple OneShotAgent optimized for simple tool calling tasks.

This agent invokes the OneShotToolCallingModel up to four times, but each individual
attempt is a one-shot call. It is useful when the tool call is simple, minimizing cost.
However, for more complex tool calls, the DefaultExecutionAgent is recommended as it will
be more successful than the OneShotAgent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from portia.config import EXECUTION_MODEL_KEY
from portia.errors import InvalidAgentError
from portia.execution_agents.base_execution_agent import BaseExecutionAgent
from portia.execution_agents.execution_utils import (
    AgentNode,
    next_state_after_tool_call,
    process_output,
    tool_call_or_end,
)
from portia.execution_agents.utils.step_summarizer import StepSummarizer
from portia.execution_context import get_execution_context
from portia.tool import ToolRunContext

if TYPE_CHECKING:
    from langchain.tools import StructuredTool

    from portia.config import Config
    from portia.execution_agents.output import Output
    from portia.model import LangChainGenerativeModel
    from portia.plan import Step
    from portia.plan_run import PlanRun
    from portia.tool import Tool


class OneShotToolCallingModel:
    """One-shot model for calling a given tool.

    This model directly passes the tool and context to the language model (LLM)
    to generate a response. It is suitable for simple tasks where the arguments
    are already correctly formatted and complete. This model does not validate
    arguments (e.g., it will not catch missing arguments).

    It is recommended to use the DefaultExecutionAgent for more complex tasks.

    Args:
        model (GenerativeModel): The language model to use for generating responses.
        context (str): The context to provide to the language model when generating a response.
        tools (list[StructuredTool]): A list of tools that can be used during the task.
        agent (OneShotAgent): The agent responsible for managing the task.

    Methods:
        invoke(MessagesState): Invokes the LLM to generate a response based on the query, context,
                               and past errors.

    """

    tool_calling_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are very powerful assistant, but don't know current events.",
            ),
            HumanMessagePromptTemplate.from_template(
                [
                    "query:",
                    "{query}",
                    "context:",
                    "{context}",
                    "Use the provided tool. You should provide arguments that match the tool's"
                    "schema using the information contained in the query and context."
                    "Make sure you don't repeat past errors: {past_errors}",
                ],
            ),
        ],
    )

    def __init__(
        self,
        model: LangChainGenerativeModel,
        context: str,
        tools: list[StructuredTool],
        agent: OneShotAgent,
    ) -> None:
        """Initialize the OneShotToolCallingModel.

        Args:
            model (LangChainGenerativeModel): The language model to use for generating responses.
            context (str): The context to be used when generating the response.
            tools (list[StructuredTool]): A list of tools that can be used during the task.
            agent (OneShotAgent): The agent that is managing the task.

        """
        self.model = model
        self.context = context
        self.agent = agent
        self.tools = tools

    def invoke(self, state: MessagesState) -> dict[str, Any]:
        """Invoke the model with the given message state.

        This method formats the input for the language model using the query, context,
        and past errors, then generates a response by invoking the model.

        Args:
            state (MessagesState): The state containing the messages and other necessary data.

        Returns:
            dict[str, Any]: A dictionary containing the model's generated response.

        """
        model = self.model.to_langchain().bind_tools(self.tools)
        messages = state["messages"]
        past_errors = [msg for msg in messages if "ToolSoftError" in msg.content]
        response = model.invoke(
            self.tool_calling_prompt.format_messages(
                query=self.agent.step.task,
                context=self.context,
                past_errors=past_errors,
            ),
        )
        return {"messages": [response]}


class OneShotAgent(BaseExecutionAgent):
    """Agent responsible for achieving a task by using langgraph.

    This agent performs the following steps:
    1. Calls the tool with unverified arguments.
    2. Retries tool calls up to 4 times.

    Args:
        step (Step): The current step in the task plan.
        plan_run (PlanRun): The run that defines the task execution process.
        config (Config): The configuration settings for the agent.
        tool (Tool | None): The tool to be used for the task (optional).

    Methods:
        execute_sync(): Executes the core logic of the agent's task, using the provided tool

    """

    def __init__(
        self,
        step: Step,
        plan_run: PlanRun,
        config: Config,
        tool: Tool | None = None,
    ) -> None:
        """Initialize the OneShotAgent.

        Args:
            step (Step): The current step in the task plan.
            plan_run (PlanRun): The run that defines the task execution process.
            config (Config): The configuration settings for the agent.
            tool (Tool | None): The tool to be used for the task (optional).

        """
        super().__init__(step, plan_run, config, tool)

    def execute_sync(self) -> Output:
        """Run the core execution logic of the task.

        This method will invoke the tool with arguments

        Returns:
            Output: The result of the agent's execution, containing the tool call result.

        """
        if not self.tool:
            raise InvalidAgentError("No tool available")

        context = self.get_system_context()
        model = self.config.resolve_langchain_model(EXECUTION_MODEL_KEY)
        tools = [
            self.tool.to_langchain_with_artifact(
                ctx=ToolRunContext(
                    execution_context=get_execution_context(),
                    plan_run_id=self.plan_run.id,
                    config=self.config,
                    clarifications=self.plan_run.get_clarifications_for_step(),
                ),
            ),
        ]
        tool_node = ToolNode(tools)

        graph = StateGraph(MessagesState)
        graph.add_node(
            AgentNode.TOOL_AGENT,
            OneShotToolCallingModel(model, context, tools, self).invoke,
        )
        graph.add_node(AgentNode.TOOLS, tool_node)
        graph.add_node(
            AgentNode.SUMMARIZER,
            StepSummarizer(self.config, model, self.tool, self.step).invoke,
        )
        graph.add_edge(START, AgentNode.TOOL_AGENT)

        # Use execution manager for state transitions
        graph.add_conditional_edges(
            AgentNode.TOOL_AGENT,
            tool_call_or_end,
        )
        graph.add_conditional_edges(
            AgentNode.TOOLS,
            lambda state: next_state_after_tool_call(self.config, state, self.tool),
        )
        graph.add_edge(AgentNode.SUMMARIZER, END)

        app = graph.compile()
        invocation_result = app.invoke({"messages": []})

        return process_output(invocation_result["messages"], self.tool)
