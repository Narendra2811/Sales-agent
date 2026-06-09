from abc import ABC, abstractmethod  # ABC = Abstract Base Class
from typing import Optional
import datetime


class MemoryBackend(ABC):
    """
    Abstract base class for memory storage backends.

    """

    @abstractmethod
    def save_message(
        self, user_id: str, session_id: str, role: str, content: str
    ) -> None:
        """
        Save a single message to the conversation history.

        Args:
            user_id:    Identifier for the user (from the URL path)
            session_id: UUID for this conversation session
            role:       "user" or "assistant" (who sent the message)
            content:    The actual text of the message
        """
        ...

    @abstractmethod
    def get_recent_messages(self, user_id: str, limit: int = 10) -> list[dict]:
        """
        Retrieve the most recent messages for a user (short-term memory).

        Returns messages in CHRONOLOGICAL order (oldest first, newest last).
        This order is important because the LLM needs context in the right sequence.

        Args:
            user_id: Which user's messages to retrieve
            limit:   Maximum number of messages to return

        Returns:
            List of dicts like: [{"role": "user", "content": "..."}, ...]
        """
        ...

    @abstractmethod
    def get_full_history(self, user_id: str) -> list[dict]:
        """
        Retrieve ALL messages for a user across all sessions.
        Used by the GET /chat/{user_id}/history endpoint.

        Returns messages in chronological order.
        """
        ...

    @abstractmethod
    def get_message_count(self, user_id: str) -> int:
        """
        Count how many total messages a user has.
        Used to decide when to trigger auto-summarization.

        If count > SUMMARIZATION_THRESHOLD -> trigger summarize_memory.
        """
        ...

    @abstractmethod
    def save_summary(
        self,
        user_id: str,
        session_id: str,
        summary_text: str,
        message_count_covered: int,
    ) -> None:
        """
        Save a compressed summary of older conversation history.

        Called by the summarization trigger in the save_and_eval node
        when message_count > SUMMARIZATION_THRESHOLD.

        Args:
            user_id:               Which user this summary is for
            session_id:            Session during which summarization happened
            summary_text:          The compressed summary paragraph from the model
            message_count_covered: How many messages are covered by this summary
        """
        ...

    @abstractmethod
    def get_latest_summary(self, user_id: str) -> Optional[str]:
        """
        Get the most recent summary for a user.

        Returns the summary TEXT (a paragraph), or None if no summary exists yet.
        The summary + last K raw messages = full short-term context.
        """
        ...

    @abstractmethod
    def save_user_fact(
        self, user_id: str, fact_key: str, fact_value: str, confidence: float = 1.0
    ) -> None:
        """
        Save or update a long-term fact about a user.

        Uses UPSERT logic:
         
        Called by the fact extractor in the eval service after every turn.

        Args:
            user_id:    Which user this fact is about
            fact_key:   Category: "team_size", "budget", "pain_point", etc.
            fact_value: The value: "50", "$500/mo", "needs SSO", etc.
            confidence: How certain we are (0.0 to 1.0). Default 1.0 = certain.
        """
        ...

    @abstractmethod
    def get_user_facts(self, user_id: str) -> list[dict]:
        """
        Get all stored long-term facts for a user.

        Returns a list of fact dicts:
        [
            {"key": "team_size", "value": "50", "confidence": 1.0},
            {"key": "budget", "value": "$500/mo", "confidence": 0.9},
        ]
        """
        ...

    @abstractmethod
    def get_user_facts_string(self, user_id: str) -> str:
        """
        Get all user facts formatted as a readable string for LLM injection.

        Example output:
            "team_size: 50
             budget: $500/mo
             pain_point: needs SSO for enterprise clients
             plan_interest: Enterprise"

        This string is injected into the agent's system prompt.
        """
        ...

    @abstractmethod
    def get_memory_context_string(self, user_id: str, recent_limit: int = 10) -> str:
        """
        Assembles the COMPLETE short-term context string for the agent.

        """
        ...

    @abstractmethod
    def delete_user_memory(self, user_id: str) -> dict:
        """
        Permanently delete ALL memory data for a user.
        """
        ...

    @abstractmethod
    def create_session(self, session_id: str, user_id: str) -> None:
        """
        Create a new session record in the database.
        Called at the start of every POST /chat request.
        """
        ...

    @abstractmethod
    def close_session(self, session_id: str, message_count: int) -> None:
        """
        Mark a session as ended and record the message count.
        Called at the end of every POST /chat request (in save_and_eval).
        """
        ...
