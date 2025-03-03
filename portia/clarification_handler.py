"""Clarification Handler.

This module defines the base ClarificationHandler interface that determines how to handle
clarifications that arise during the execution of a workflow. It also provides a
CLIClarificationHandler implementation that handles clarifications via the CLI.
"""

from __future__ import annotations

from abc import ABC
from typing import Callable

from portia.clarification import (
    ActionClarification,
    Clarification,
    CustomClarification,
    InputClarification,
    MultipleChoiceClarification,
    ValueConfirmationClarification,
)


class ClarificationHandler(ABC):  # noqa: B024
    """Handles clarifications that arise during the execution of a plan run."""

    def handle(
        self,
        clarification: Clarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle a clarification by routing it to the appropriate handler.

        Args:
            clarification: The clarification object to handle
            on_resolution: Callback function that should be invoked once the clarification has been
                handled, prompting the plan run to resume. This can either be called synchronously
                in this function or called async after returning from this function.
            on_error: Callback function to mark that should be invokved if the clarification handling
                has failed. This can either be called synchronously in this function or called async
                after returning from this function.

        """
        match clarification:
            case ActionClarification():
                f = self.handle_action_clarification
            case InputClarification():
                f = self.handle_input_clarification
            case MultipleChoiceClarification():
                f = self.handle_multiple_choice_clarification
            case ValueConfirmationClarification():
                f = self.handle_value_confirmation_clarification
            case CustomClarification():
                f = self.handle_custom_clarification
            case _:
                raise ValueError(
                    f"Attempted to handle an unknown clarification type: {type(clarification)}",
                )
        f(clarification, on_resolution, on_error)

    def handle_action_clarification(
        self,
        clarification: ActionClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle an action clarification."""
        raise NotImplementedError("handle_action_clarification is not implemented")

    def handle_input_clarification(
        self,
        clarification: InputClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle a user input clarification."""
        raise NotImplementedError("handle_input_clarification is not implemented")

    def handle_multiple_choice_clarification(
        self,
        clarification: MultipleChoiceClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle a multi-choice clarification."""
        raise NotImplementedError("handle_multiple_choice_clarification is not implemented")

    def handle_value_confirmation_clarification(
        self,
        clarification: ValueConfirmationClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle a value confirmation clarification."""
        raise NotImplementedError("handle_value_confirmation_clarification is not implemented")

    def handle_custom_clarification(
        self,
        clarification: CustomClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle a custom clarification."""
        raise NotImplementedError("handle_custom_clarification is not implemented")
