from typing import Any, Dict

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
    """Handles clarifications that arise during the execution of a workflow."""

    def handle(self, clarification: Clarification) -> Workflow:
        """Routes the clarification to the appropriate handler based on its category.

        Args:
            clarification: The clarification object to handle

        Returns:
            Dict containing the handler response

        Raises:
            ValueError: If no handler exists for the clarification category

        """
        match clarification.category:
            case ClarificationCategory.ARGUMENT:
                return self.handle_argument(clarification)
            case ClarificationCategory.ACTION:
                return self.handle_action(clarification)
            case ClarificationCategory.INPUT:
                return self.handle_input(clarification)
            case ClarificationCategory.MULTIPLE_CHOICE:
                return self.handle_multiple_choice(clarification)
            case ClarificationCategory.VALUE_CONFIRMATION:
                return self.handle_value_confirmation(clarification)
            case ClarificationCategory.CUSTOM:
                return self.handle_custom(clarification)

    def handle_input(self, clarification: InputClarification) -> Dict[str, Any]:
        """Handles input clarifications where user needs to provide a value.

        Args:
            clarification: The input clarification

        Returns:
            Dict containing the user's input value

        """
        # Implementation for handling input clarifications
        raise NotImplementedError("Input clarification handling not implemented")

    def handle_action(self, clarification: ActionClarification) -> Dict[str, Any]:
        """Handles action clarifications that require user to complete an action (e.g. clicking a URL).

        Args:
            clarification: The action clarification

        Returns:
            Dict containing the action completion status

        """
        # Implementation for handling action clarifications
        raise NotImplementedError("Action clarification handling not implemented")

    def handle_multiple_choice(self, clarification: MultipleChoiceClarification) -> Dict[str, Any]:
        """Handles multiple choice clarifications where user must select from options.

        Args:
            clarification: The multiple choice clarification

        Returns:
            Dict containing the selected option

        """
        # Implementation for handling multiple choice clarifications
        raise NotImplementedError("Multiple choice clarification handling not implemented")

    def handle_value_confirmation(
        self,
        clarification: ValueConfirmationClarification,
    ) -> Dict[str, Any]:
        """Handles value confirmation clarifications where user must confirm a value.

        Args:
            clarification: The value confirmation clarification

        Returns:
            Dict containing the confirmation status

        """
        # Implementation for handling value confirmation clarifications
        raise NotImplementedError("Value confirmation clarification handling not implemented")

    def handle_custom(self, clarification: CustomClarification) -> Dict[str, Any]:
        """Handles custom clarifications with arbitrary data.

        Args:
            clarification: The custom clarification

        Returns:
            Dict containing the custom clarification response

        """
        # Implementation for handling custom clarifications
        raise NotImplementedError("Custom clarification handling not implemented")
