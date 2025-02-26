"""Clarification Handler.

This module defines the base ClarificationHandler interface that determines how to handle
clarifications that arise during the execution of a workflow. It also provides a
CLIClarificationHandler implementation that handles clarifications via the CLI.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Callable

from portia.clarification import (
    ActionClarification,
    Clarification,
    ClarificationCategory,
    CustomClarification,
    InputClarification,
    MultipleChoiceClarification,
    ValueConfirmationClarification,
)


class ClarificationHandler:
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
        match type(clarification):
            case ClarificationCategory.ACTION:
                return self.handle_action_clarification(
                    clarification,
                    resolve,
                    error,
                )
            case ClarificationCategory.INPUT:
                return self.handle_input_clarification(
                    clarification,
                    resolve,
                    error,
                )
            case ClarificationCategory.MULTIPLE_CHOICE:
                return self.handle_multiple_choice_clarification(
                    clarification,
                    resolve,
                    error,
                )
            case ClarificationCategory.VALUE_CONFIRMATION:
                return self.handle_value_confirmation_clarification(
                    clarification,
                    resolve,
                    error,
                )
            case ClarificationCategory.CUSTOM:
                return self.handle_custom_clarification(
                    clarification,
                    resolve,
                    error,
                )

    @abstractmethod
    def handle_action_clarification(
        self,
        clarification: ActionClarification,
        resolve: Callable[[Clarification, object], None],
        error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle an action clarification."""

    @abstractmethod
    def handle_input_clarification(
        self,
        clarification: InputClarification,
        resolve: Callable[[Clarification, object], None],
        error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle a user input clarification."""

    @abstractmethod
    def handle_multiple_choice_clarification(
        self,
        clarification: MultipleChoiceClarification,
        resolve: Callable[[Clarification, object], None],
        error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle a multi-choice clarification."""

    @abstractmethod
    def handle_value_confirmation_clarification(
        self,
        clarification: ValueConfirmationClarification,
        resolve: Callable[[Clarification, object], None],
        error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle a value confirmation clarification."""

    @abstractmethod
    def handle_custom_clarification(
        self,
        clarification: CustomClarification,
        resolve: Callable[[Clarification, object], None],
        error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle a custom clarification."""
