"""TODO"""

import json

import click

from portia.clarification import (
    ActionClarification,
    CustomClarification,
    InputClarification,
    MultipleChoiceClarification,
    ValueConfirmationClarification,
)
from portia.logger import logger
from portia.runner import Runner
from portia.workflow import Workflow, WorkflowState


class DefaultClarificationHandler:
    """A default clarification handler that allows the user to handle clarifications on the CLI."""

    def handle_argument_clarification(
        self,
        runner: Runner,
        workflow: Workflow,
        clarification: ActionClarification,
    ) -> Workflow:
        """Present the action clarification to the user on the CLI."""
        logger().info(
            f"{clarification.user_guidance} -- Please click on the link below to proceed.",
            f"{clarification.action_url}",
        )
        return runner.wait_for_ready(workflow)

    def handle_action_clarification(
        self,
        runner: Runner,
        workflow: Workflow,
        clarification: ActionClarification,
    ) -> Workflow:
        """Handle a clarification that needs the user to complete an action (e.g. click a URL)."""
        logger().info(
            f"{clarification.user_guidance} -- Please click on the link below to proceed.",
            f"{clarification.action_url}",
        )
        return runner.wait_for_ready(workflow)

    def handle_input_clarification(
        self,
        runner: Runner,
        workflow: Workflow,
        clarification: InputClarification,
    ) -> Workflow:
        """Handle a user input clarifications by asking the user for input from the CLI."""
        user_input = click.prompt(
            clarification.user_guidance + "\nPlease enter a value:\n",
        )
        return runner.resolve_clarification(clarification, user_input, workflow)

    def handle_multiple_choice_clarification(
        self,
        runner: Runner,
        workflow: Workflow,
        clarification: MultipleChoiceClarification,
    ) -> Workflow:
        """Handle a multi-choice clarification by asking the user for input from the CLI."""
        choices = click.Choice(clarification.options)
        user_input = click.prompt(
            clarification.user_guidance + "\nPlease choose a value:\n",
            type=choices,
        )
        return runner.resolve_clarification(clarification, user_input, workflow)

    def handle_value_confirmation_clarification(
        self,
        runner: Runner,
        workflow: Workflow,
        clarification: ValueConfirmationClarification,
    ) -> Workflow:
        """Ask the user to confirm the value on the CLI."""
        if click.confirm(text=clarification.user_guidance, default=False):
            return runner.resolve_clarification(
                clarification,
                response=True,
                workflow=workflow,
            )
        workflow.state = WorkflowState.FAILED
        runner.storage.save_workflow(workflow)
        return workflow

    def handle_custom_clarification(
        self,
        runner: Runner,
        workflow: Workflow,
        clarification: CustomClarification,
    ) -> Workflow:
        """Handle a custom clarification by presenting it to the user on the CLI."""
        click.echo(clarification.user_guidance)
        click.echo(f"Additional data: {json.dumps(clarification.data)}")
        user_input = click.prompt("\nPlease enter a value:\n")
        return runner.resolve_clarification(clarification, user_input, workflow)
