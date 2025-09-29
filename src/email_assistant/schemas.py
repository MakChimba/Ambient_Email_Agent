from typing import Any, Dict, List

from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal
from langgraph.graph import MessagesState

class RouterSchema(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )

class StateInput(TypedDict):
    # This is the input to the state
    email_input: dict

class State(MessagesState):
    # This state class has the messages key built in
    email_input: dict
    classification_decision: Literal["ignore", "respond", "notify"]
    # Optional fields populated at finalization for evaluators/Studio mapping
    assistant_reply: str | None = None
    tool_trace: str | None = None
    email_markdown: str | None = None
    # Reminder workflow coordination fields
    reminder_actions: List[Dict[str, Any]] | None = None
    reminder_thread_id: str | None = None
    reminder_next_node: str | None = None
    reminder_dispatch_origin: str | None = None
    reminder_dispatch_outcome: Dict[str, Any] | None = None

class EmailData(TypedDict):
    id: str
    thread_id: str
    from_email: str
    subject: str
    page_content: str
    send_time: str
    to_email: str

class UserPreferences(BaseModel):
    """Updated user preferences based on user's feedback."""

    rationale: str = Field(
        description="Brief rationale for updating user preferences based on feedback."
    )
    user_preferences: str = Field(description="Updated user preferences")


class AssistantContext(TypedDict, total=False):
    """Runtime context propagated through LangGraph Runtime."""

    timezone: str
    eval_mode: bool
    thread_id: str | None
    thread_metadata: Dict[str, Any]
