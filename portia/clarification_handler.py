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
        resolve: Callable[[Clarification, object], None],
        error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle a clarification by routing it to the appropriate handler.

        Args:
            clarification: The clarification object to handle
            resolve: Callback function to resolve the clarification. This can either be called
                synchronously in this function or called async after returning from this function.
            error: Callback function to mark that handlign a clarification has failed. This can
                either be called synchronously in this function or called async after returning
                from this function.

        """
        match clarification:
            case ActionClarification():
                return self.handle_action_clarification(
                    clarification,
                    resolve,
                    error,
                )
            case InputClarification():
                return self.handle_input_clarification(
                    clarification,
                    resolve,
                    error,
                )
            case MultipleChoiceClarification():
                return self.handle_multiple_choice_clarification(
                    clarification,
                    resolve,
                    error,
                )
            case ValueConfirmationClarification():
                return self.handle_value_confirmation_clarification(
                    clarification,
                    resolve,
                    error,
                )
            case CustomClarification():
                return self.handle_custom_clarification(
                    clarification,
                    resolve,
                    error,
                )
            case _:
                raise ValueError(
                    f"Attempted to handle an unknown clarification type: {type(clarification)}",
                )

    def handle_action_clarification(
        self,
        clarification: ActionClarification,
        resolve: Callable[[Clarification, object], None],
        error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle an action clarification."""
        raise NotImplementedError("handle_action_clarification is not implemented")

    def handle_input_clarification(
        self,
        clarification: InputClarification,
        resolve: Callable[[Clarification, object], None],
        error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle a user input clarification."""
        raise NotImplementedError("handle_input_clarification is not implemented")

    def handle_multiple_choice_clarification(
        self,
        clarification: MultipleChoiceClarification,
        resolve: Callable[[Clarification, object], None],
        error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle a multi-choice clarification."""
        raise NotImplementedError("handle_multiple_choice_clarification is not implemented")

    def handle_value_confirmation_clarification(
        self,
        clarification: ValueConfirmationClarification,
        resolve: Callable[[Clarification, object], None],
        error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle a value confirmation clarification."""
        raise NotImplementedError("handle_value_confirmation_clarification is not implemented")

    def handle_custom_clarification(
        self,
        clarification: CustomClarification,
        resolve: Callable[[Clarification, object], None],
        error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle a custom clarification."""
        raise NotImplementedError("handle_custom_clarification is not implemented")
