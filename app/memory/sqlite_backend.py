import logging
import datetime
from typing import Optional

from sqlalchemy.orm import (
    Session as DBSession,
)  # Renamed to avoid conflict with our Session model

from app.memory.base import MemoryBackend
from app.db.models import Message, UserFact, Summary, Session, Flag, Eval
from app.config import settings

logger = logging.getLogger(__name__)


class SQLiteMemoryBackend(MemoryBackend):
    """
    Memory backend implementation using SQLite via SQLAlchemy.


    """

    def __init__(self, db: DBSession):
        """
        Args:
            db: An active SQLAlchemy session.
                All database operations use this session.
                The caller is responsible for committing and closing it.
        """
        self.db = db

    def save_message(
        self, user_id: str, session_id: str, role: str, content: str
    ) -> None:
        """
        Save one message to the messages table.

        This is called twice per conversation turn:
          1. When user sends a message (role="user")
          2. When agent sends a response (role="assistant")

        Example:
            backend.save_message("alice", "uuid-123", "user", "What's Enterprise pricing?")
        """
        try:
            message = Message(
                user_id=user_id,
                session_id=session_id,
                role=role,
                content=content,
                created_at=datetime.datetime.utcnow(),
            )
            self.db.add(message)  # Stage the INSERT
            self.db.commit()  # Actually write to disk
            logger.debug(f"Saved message for user={user_id} role={role}")
        except Exception as e:
            self.db.rollback()  # Undo if something went wrong
            logger.error(f"Failed to save message for user={user_id}: {e}")
            raise

    def get_recent_messages(self, user_id: str, limit: int = 10) -> list[dict]:
        """
        Get the N most recent messages for a user, in chronological order.

        Used for short-term context injection into the agent.
        Returns only "user" and "assistant" messages (not tool messages).


        Example return value:
            [
                {"role": "user", "content": "What's Enterprise pricing?"},
                {"role": "assistant", "content": "Enterprise is $499/mo..."},
                {"role": "user", "content": "Does that include SSO?"},
            ]
        """
        try:
            rows = (
                self.db.query(Message)
                .filter(
                    Message.user_id == user_id, Message.role.in_(["user", "assistant"])
                )
                .order_by(Message.created_at.desc())  # Newest first
                .limit(limit)
                .all()
            )
            rows.reverse()

            return [{"role": row.role, "content": row.content} for row in rows]

        except Exception as e:
            logger.error(f"Failed to get recent messages for user={user_id}: {e}")
            return []

    def get_full_history(self, user_id: str) -> list[dict]:
        """
        Get ALL messages for a user across all sessions, in chronological order.
        Used by the GET /chat/{user_id}/history API endpoint.

        Returns the full ORM objects (not just dicts) so the API can include
        timestamps, session_ids, and message IDs.
        """
        try:
            rows = (
                self.db.query(Message)
                .filter(
                    Message.user_id == user_id, Message.role.in_(["user", "assistant"])
                )
                .order_by(Message.created_at.asc())  # Oldest first
                .all()
            )
            return rows  # Returns list of Message ORM objects
        except Exception as e:
            logger.error(f"Failed to get full history for user={user_id}: {e}")
            return []

    def get_message_count(self, user_id: str) -> int:
        """
        Count total messages (user + assistant) for a user.
        Used to decide whether to trigger auto-summarization.

        If this count > settings.SUMMARIZATION_THRESHOLD -> summarize old messages.
        """
        try:
            count = (
                self.db.query(Message)
                .filter(
                    Message.user_id == user_id, Message.role.in_(["user", "assistant"])
                )
                .count()
            )
            return count
        except Exception as e:
            logger.error(f"Failed to count messages for user={user_id}: {e}")
            return 0

    def save_summary(
        self,
        user_id: str,
        session_id: str,
        summary_text: str,
        message_count_covered: int,
    ) -> None:
        """
        Save a compressed summary of older conversations.

        When this is saved, the next request will inject this summary
        (instead of all the old raw messages) as context for the agent.
        This keeps the token count bounded while preserving key information.
        """
        try:
            summary = Summary(
                user_id=user_id,
                session_id=session_id,
                summary_text=summary_text,
                message_count_covered=message_count_covered,
                created_at=datetime.datetime.utcnow(),
            )
            self.db.add(summary)
            self.db.commit()
            logger.info(
                f"Saved summary for user={user_id} "
                f"covering {message_count_covered} messages"
            )
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to save summary for user={user_id}: {e}")
            raise

    def get_latest_summary(self, user_id: str) -> Optional[str]:
        """
        Get the most recent summary paragraph for a user.

        Returns just the text string, or None if no summary exists yet.


        """
        try:
            summary = (
                self.db.query(Summary)
                .filter(Summary.user_id == user_id)
                .order_by(Summary.created_at.desc())  # Most recent first
                .first()
            )
            return summary.summary_text if summary else None
        except Exception as e:
            logger.error(f"Failed to get summary for user={user_id}: {e}")
            return None

    def save_user_fact(
        self, user_id: str, fact_key: str, fact_value: str, confidence: float = 1.0
    ) -> None:
        """
        Save or UPDATE a long-term fact about a user (upsert logic).



        Example:
            backend.save_user_fact("alice", "team_size", "20")

            backend.save_user_fact("alice", "team_size", "50")
        """
        try:
            existing = (
                self.db.query(UserFact)
                .filter(UserFact.user_id == user_id, UserFact.fact_key == fact_key)
                .first()
            )

            if existing:
                existing.fact_value = fact_value
                existing.confidence = confidence
                existing.updated_at = datetime.datetime.utcnow()
                logger.debug(
                    f"Updated fact for user={user_id}: {fact_key}='{fact_value}'"
                )
            else:
                fact = UserFact(
                    user_id=user_id,
                    fact_key=fact_key,
                    fact_value=fact_value,
                    confidence=confidence,
                    created_at=datetime.datetime.utcnow(),
                    updated_at=datetime.datetime.utcnow(),
                )
                self.db.add(fact)
                logger.debug(
                    f"Inserted new fact for user={user_id}: {fact_key}='{fact_value}'"
                )

            self.db.commit()

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to save user fact for user={user_id}: {e}")
            raise

    def get_user_facts(self, user_id: str) -> list[dict]:
        """
        Get all long-term facts for a user as a list of dicts.

        Returns:
            [
                {"key": "team_size", "value": "50", "confidence": 1.0},
                {"key": "budget", "value": "$500/mo", "confidence": 0.9},
                {"key": "pain_point", "value": "needs SSO", "confidence": 1.0},
            ]
        """
        try:
            facts = (
                self.db.query(UserFact)
                .filter(UserFact.user_id == user_id)
                .order_by(UserFact.fact_key)  # Alphabetical order for readability
                .all()
            )
            return [
                {
                    "key": f.fact_key,
                    "value": f.fact_value,
                    "confidence": f.confidence or 1.0,
                }
                for f in facts
            ]
        except Exception as e:
            logger.error(f"Failed to get user facts for user={user_id}: {e}")
            return []

    def get_user_facts_string(self, user_id: str) -> str:
        """
        Get all facts formatted as a readable string for LLM injection.

        If no facts exist, returns an empty string (agent handles gracefully).

        Example output:
            "budget: $500/mo
             company_name: Acme Corp
             pain_point: needs SSO for enterprise clients
             team_size: 50"
        """
        facts = self.get_user_facts(user_id)

        if not facts:
            return ""  # No known facts yet

        lines = [f"  {f['key']}: {f['value']}" for f in facts]
        return "\n".join(lines)

    def get_user_context_string(self, user_id: str) -> str:
        """
        Get complete user context as a string (facts + recent messages).
        This is what the get_user_memory tool returns to the LLM.

        Combines:
          - Long-term facts (always included)
          - Most recent messages (for immediate context)

        This gives the LLM a complete picture of what we know about this user.
        """
        facts_str = self.get_user_facts_string(user_id)
        recent = self.get_recent_messages(user_id, limit=5)

        result_parts = []

        if facts_str:
            result_parts.append(f"Known facts about this user:\n{facts_str}")
        else:
            result_parts.append("No facts recorded yet for this user.")

        if recent:
            msg_lines = [f"  {m['role']}: {m['content'][:200]}" for m in recent]
            result_parts.append(
                f"Last {len(recent)} messages:\n" + "\n".join(msg_lines)
            )

        return "\n\n".join(result_parts)

    def get_memory_context_string(self, user_id: str, recent_limit: int = 10) -> str:
        """
                Assembles the complete short-term context string for injection into the agent.
        ."
        """
        summary = self.get_latest_summary(user_id)
        recent_messages = self.get_recent_messages(user_id, limit=recent_limit)

        parts = []

        if summary:
            parts.append(f"=== Previous Conversation Summary ===\n{summary}")

        if recent_messages:
            msg_lines = [f"{m['role']}: {m['content']}" for m in recent_messages]
            parts.append("=== Recent Conversation ===\n" + "\n".join(msg_lines))

        if not parts:
            return ""  # Brand new user, no history at all

        return "\n\n".join(parts)

    def save_eval(
        self,
        user_id: str,
        session_id: str,
        groundedness: float,
        relevance: float,
        confidence: float,
        flagged: bool,
        reasoning: str,
        tools_called: list[str],
    ) -> None:
        """
        Save an evaluation record for a single agent response.

        Called in save_and_eval_node after every agent response.
        The data stored here powers the GET /chat/{user_id}/evals endpoint.
        """
        import json

        try:
            eval_record = Eval(
                user_id=user_id,
                session_id=session_id,
                groundedness=groundedness,
                relevance=relevance,
                confidence=confidence,
                flagged=flagged,
                reasoning=reasoning,
                tools_called=json.dumps(tools_called),  # Serialize list -> JSON string
                created_at=datetime.datetime.utcnow(),
            )
            self.db.add(eval_record)
            self.db.commit()
            logger.debug(f"Saved eval for user={user_id} session={session_id[:8]}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to save eval for user={user_id}: {e}")
            raise

    def get_eval_aggregate(self, user_id: str) -> dict:
        """
        Calculate aggregate eval statistics for a user across all sessions.

        Used by the bonus GET /chat/{user_id}/evals endpoint.

        Returns a dict with:
          - avg_groundedness, avg_relevance, avg_confidence
          - pct_flagged (percentage of responses flagged)
          - total_responses, total_flagged
          - recent_evals (last 10 individual eval records)
        """
        import json

        try:
            all_evals = (
                self.db.query(Eval)
                .filter(Eval.user_id == user_id)
                .order_by(Eval.created_at.desc())
                .all()
            )

            if not all_evals:
                return {
                    "total_responses": 0,
                    "avg_groundedness": 0.0,
                    "avg_relevance": 0.0,
                    "avg_confidence": 0.0,
                    "pct_flagged": 0.0,
                    "total_flagged": 0,
                    "recent_evals": [],
                }

            total = len(all_evals)
            flagged_count = sum(1 for e in all_evals if e.flagged)

            avg_ground = sum(e.groundedness for e in all_evals) / total
            avg_rel = sum(e.relevance for e in all_evals) / total
            avg_conf = sum(e.confidence for e in all_evals) / total

            recent = all_evals[:10]
            recent_formatted = [
                {
                    "session_id": e.session_id,
                    "groundedness": e.groundedness,
                    "relevance": e.relevance,
                    "confidence": e.confidence,
                    "flagged": e.flagged,
                    "reasoning": e.reasoning,
                    "tools_called": json.loads(e.tools_called or "[]"),
                    "created_at": e.created_at,
                }
                for e in recent
            ]

            return {
                "total_responses": total,
                "avg_groundedness": round(avg_ground, 3),
                "avg_relevance": round(avg_rel, 3),
                "avg_confidence": round(avg_conf, 3),
                "pct_flagged": round((flagged_count / total) * 100, 1),
                "total_flagged": flagged_count,
                "recent_evals": recent_formatted,
            }

        except Exception as e:
            logger.error(f"Failed to compute eval aggregate for user={user_id}: {e}")
            return {}

    def create_session(self, session_id: str, user_id: str) -> None:
        """Create a new session record at the start of a chat request."""
        try:
            session = Session(
                id=session_id,
                user_id=user_id,
                started_at=datetime.datetime.utcnow(),
                message_count=0,
            )
            self.db.add(session)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            logger.warning(f"Failed to create session {session_id}: {e}")

    def close_session(self, session_id: str, message_count: int) -> None:
        """Mark a session as ended with final message count."""
        try:
            session = self.db.query(Session).filter(Session.id == session_id).first()
            if session:
                session.ended_at = datetime.datetime.utcnow()
                session.message_count = message_count
                self.db.commit()
        except Exception as e:
            self.db.rollback()
            logger.warning(f"Failed to close session {session_id}: {e}")

    def delete_user_memory(self, user_id: str) -> dict:
        """
        Permanently delete ALL data for a user
        """
        try:
            deleted_counts = {}

            deleted_counts["messages"] = (
                self.db.query(Message)
                .filter(Message.user_id == user_id)
                .delete(
                    synchronize_session=False
                )  # Don't load into memory, just DELETE
            )

            deleted_counts["user_facts"] = (
                self.db.query(UserFact)
                .filter(UserFact.user_id == user_id)
                .delete(synchronize_session=False)
            )

            deleted_counts["summaries"] = (
                self.db.query(Summary)
                .filter(Summary.user_id == user_id)
                .delete(synchronize_session=False)
            )

            deleted_counts["sessions"] = (
                self.db.query(Session)
                .filter(Session.user_id == user_id)
                .delete(synchronize_session=False)
            )

            deleted_counts["evals"] = (
                self.db.query(Eval)
                .filter(Eval.user_id == user_id)
                .delete(synchronize_session=False)
            )

            deleted_counts["flags"] = (
                self.db.query(Flag)
                .filter(Flag.user_id == user_id)
                .delete(synchronize_session=False)
            )

            self.db.commit()

            logger.info(
                f"Deleted all data for user={user_id}: "
                f"{sum(deleted_counts.values())} total rows"
            )
            return deleted_counts

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to delete memory for user={user_id}: {e}")
            raise
