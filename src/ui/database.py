"""Database module for managing chat conversations with PostgreSQL."""

import os
import uuid
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.base import BaseMessage
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

# Load environment variables
load_dotenv()

Base = declarative_base()  # type: ignore[misc]


def _serialize_messages(messages: list[BaseMessage]) -> list[dict]:
    """Convert LangChain messages to JSON-serializable dictionaries.

    Args:
        messages: List of LangChain message objects

    Returns:
        List of dictionaries that can be serialized to JSON
    """
    serialized = []
    for msg in messages:
        msg_dict = {
            "type": msg.__class__.__name__,
            "content": msg.content,
        }

        # Add additional fields if present
        if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
            msg_dict["additional_kwargs"] = msg.additional_kwargs  # type: ignore[assignment]
        if hasattr(msg, "response_metadata") and msg.response_metadata:
            msg_dict["response_metadata"] = msg.response_metadata  # type: ignore[assignment]
        if hasattr(msg, "id"):
            msg_dict["id"] = msg.id  # type: ignore[assignment]
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            msg_dict["tool_calls"] = msg.tool_calls
        if hasattr(msg, "name") and msg.name:
            msg_dict["name"] = msg.name

        serialized.append(msg_dict)

    return serialized


def _deserialize_messages(messages_data: list[dict[Any, Any]]) -> list[BaseMessage]:
    """Convert dictionaries back to LangChain message objects.

    Args:
        messages_data: List of message dictionaries

    Returns:
        List of LangChain message objects
    """
    messages: list[BaseMessage] = []
    for msg_dict in messages_data:
        msg_type = msg_dict.get("type", "HumanMessage")
        content = msg_dict.get("content", "")

        # Create the appropriate message type
        msg: BaseMessage
        if msg_type == "HumanMessage":
            msg = HumanMessage(content=content)
        elif msg_type == "AIMessage":
            msg = AIMessage(content=content)
        elif msg_type == "ToolMessage":
            msg = ToolMessage(
                content=content, tool_call_id=msg_dict.get("tool_call_id", "")
            )
        else:
            # Default to HumanMessage for unknown types
            msg = HumanMessage(content=content)

        # Restore additional fields if present
        if "id" in msg_dict and msg_dict["id"] is not None:
            msg.id = str(msg_dict["id"])
        if "additional_kwargs" in msg_dict:
            msg.additional_kwargs = dict(msg_dict["additional_kwargs"])
        if "response_metadata" in msg_dict:
            msg.response_metadata = dict(msg_dict["response_metadata"])
        if "tool_calls" in msg_dict and hasattr(msg, "tool_calls"):
            msg.tool_calls = msg_dict["tool_calls"]  # type: ignore[assignment]
        if "name" in msg_dict and hasattr(msg, "name") and msg_dict["name"] is not None:
            msg.name = str(msg_dict["name"])

        messages.append(msg)

    return messages


class Conversation(Base):  # type: ignore[valid-type,misc]
    """Model for storing conversation metadata."""

    __tablename__ = "conversations"

    id = Column(String, primary_key=True)  # UUID as session_id
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    message_count = Column(Integer, default=0)


class Message(Base):  # type: ignore[valid-type,misc]
    """Model for storing individual messages in conversations."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, nullable=False)  # Foreign key to conversation
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    suggested_prompts = Column(
        JSON, nullable=True
    )  # List of suggested follow-up prompts
    created_at = Column(DateTime, default=datetime.utcnow)


class ConversationState(Base):  # type: ignore[valid-type,misc]
    """Model for storing agent state as JSON."""

    __tablename__ = "conversation_states"

    conversation_id = Column(String, primary_key=True)
    agent_state = Column(JSON, nullable=False)  # Store agent messages as JSON
    config = Column(JSON, nullable=False)  # Store LangGraph config
    waiting_for_confirmation = Column(Integer, default=0)  # Boolean as int
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DatabaseManager:
    """Manager class for database operations."""

    def __init__(self, database_url: str | None = None):
        """Initialize database connection.

        Args:
            database_url: PostgreSQL connection URL. If None, uses DATABASE_URL env var
                         or falls back to SQLite for development
        """
        if database_url is None:
            database_url = os.getenv(
                "DATABASE_URL",
                "sqlite:///data/conversations.db",  # Fallback to SQLite
            )

        # Try to connect and create tables
        try:
            self.engine = create_engine(database_url, echo=False)
            self.SessionLocal = sessionmaker(bind=self.engine)

            # Create tables if they don't exist
            Base.metadata.create_all(self.engine)
        except Exception as e:
            # If PostgreSQL fails (permissions, connection, etc), fall back to SQLite
            print(
                f"Warning: Could not connect to database ({e}). Falling back to SQLite."
            )
            sqlite_url = "sqlite:///data/conversations.db"
            self.engine = create_engine(sqlite_url, echo=False)
            self.SessionLocal = sessionmaker(bind=self.engine)
            Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def create_conversation(self, name: str, session_id: str | None = None) -> str:
        """Create a new conversation.

        Args:
            name: Name of the conversation
            session_id: Optional UUID for the conversation. Auto-generated if None

        Returns:
            The conversation ID (UUID)
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        with self.get_session() as session:
            conversation = Conversation(
                id=session_id, name=name, created_at=datetime.utcnow(), message_count=0
            )
            session.add(conversation)

            # Initialize empty state
            state = ConversationState(
                conversation_id=session_id,
                agent_state={"messages": []},
                config={"configurable": {"thread_id": session_id}},
                waiting_for_confirmation=0,
            )
            session.add(state)

            session.commit()

        return session_id

    def get_all_conversations(self) -> list[dict[str, Any]]:
        """Get all conversations ordered by updated time (newest first).

        Returns:
            List of conversation dictionaries
        """
        with self.get_session() as session:
            conversations = (
                session.query(Conversation)
                .order_by(Conversation.updated_at.desc())
                .all()
            )

            return [
                {
                    "id": conv.id,
                    "name": conv.name,
                    "created_at": conv.created_at,
                    "updated_at": conv.updated_at,
                    "message_count": conv.message_count,
                }
                for conv in conversations
            ]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Get a specific conversation by ID.

        Args:
            conversation_id: The conversation UUID

        Returns:
            Conversation dictionary or None if not found
        """
        with self.get_session() as session:
            conv = session.query(Conversation).filter_by(id=conversation_id).first()
            if conv:
                return {
                    "id": conv.id,
                    "name": conv.name,
                    "created_at": conv.created_at,
                    "updated_at": conv.updated_at,
                    "message_count": conv.message_count,
                }
        return None

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        suggested_prompts: list[str] | None = None,
    ) -> None:
        """Add a message to a conversation.

        Args:
            conversation_id: The conversation UUID
            role: 'user' or 'assistant'
            content: Message content
            suggested_prompts: Optional list of suggested follow-up prompts
        """
        with self.get_session() as session:
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                suggested_prompts=suggested_prompts,
            )
            session.add(message)

            # Update conversation message count and timestamp
            conv = session.query(Conversation).filter_by(id=conversation_id).first()
            if conv:
                conv.message_count = int((conv.message_count or 0) + 1)  # type: ignore[assignment]
                conv.updated_at = datetime.utcnow()  # type: ignore[assignment]

            session.commit()

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        """Get all messages for a conversation.

        Args:
            conversation_id: The conversation UUID

        Returns:
            List of message dictionaries
        """
        with self.get_session() as session:
            messages = (
                session.query(Message)
                .filter_by(conversation_id=conversation_id)
                .order_by(Message.created_at)
                .all()
            )

            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at,
                    "suggested_prompts": msg.suggested_prompts,
                }
                for msg in messages
            ]

    def save_conversation_state(
        self,
        conversation_id: str,
        agent_state: dict,
        config: dict,
        waiting_for_confirmation: bool = False,
    ) -> None:
        """Save or update the agent state for a conversation.

        Args:
            conversation_id: The conversation UUID
            agent_state: LangGraph agent state dictionary
            config: LangGraph config dictionary
            waiting_for_confirmation: Whether waiting for user confirmation
        """
        with self.get_session() as session:
            # Serialize LangChain messages to JSON-compatible format
            serializable_state = agent_state.copy()
            if "messages" in serializable_state:
                serializable_state["messages"] = _serialize_messages(
                    serializable_state["messages"]
                )

            state = (
                session.query(ConversationState)
                .filter_by(conversation_id=conversation_id)
                .first()
            )

            if state:
                # Update existing state
                state.agent_state = serializable_state  # type: ignore[assignment]
                state.config = config  # type: ignore[assignment]
                state.waiting_for_confirmation = 1 if waiting_for_confirmation else 0  # type: ignore[assignment]
                state.updated_at = datetime.utcnow()  # type: ignore[assignment]
            else:
                # Create new state
                state = ConversationState(
                    conversation_id=conversation_id,
                    agent_state=serializable_state,
                    config=config,
                    waiting_for_confirmation=1 if waiting_for_confirmation else 0,
                )
                session.add(state)

            session.commit()

    def get_conversation_state(self, conversation_id: str) -> dict[str, Any] | None:
        """Get the agent state for a conversation.

        Args:
            conversation_id: The conversation UUID

        Returns:
            Dictionary with agent_state, config, and waiting_for_confirmation
        """
        with self.get_session() as session:
            state = (
                session.query(ConversationState)
                .filter_by(conversation_id=conversation_id)
                .first()
            )

            if state:
                # Deserialize messages back to LangChain objects
                agent_state = dict(state.agent_state)  # type: ignore[arg-type]
                messages_data = agent_state.get("messages")
                if messages_data:
                    agent_state["messages"] = _deserialize_messages(messages_data)

                return {
                    "agent_state": agent_state,
                    "config": state.config,
                    "waiting_for_confirmation": bool(state.waiting_for_confirmation),
                }
        return None

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and all its messages.

        Args:
            conversation_id: The conversation UUID
        """
        with self.get_session() as session:
            # Delete messages
            session.query(Message).filter_by(conversation_id=conversation_id).delete()

            # Delete state
            session.query(ConversationState).filter_by(
                conversation_id=conversation_id
            ).delete()

            # Delete conversation
            session.query(Conversation).filter_by(id=conversation_id).delete()

            session.commit()

    def update_conversation_name(self, conversation_id: str, name: str) -> None:
        """Update a conversation's name.

        Args:
            conversation_id: The conversation UUID
            name: New name for the conversation
        """
        with self.get_session() as session:
            conv = session.query(Conversation).filter_by(id=conversation_id).first()
            if conv:
                conv.name = name  # type: ignore[assignment]
                conv.updated_at = datetime.utcnow()  # type: ignore[assignment]
                session.commit()
