from pydantic import Field, BaseModel
from typing import Literal, Annotated, TypedDict
from langgraph.graph import add_messages, MessagesState

class UserPreferences(BaseModel):
    """Updated user preferences based on user's feedback."""
    chain_of_thought: str = Field(description="Reasoning about which user preferences need to add / update if required")
    user_preferences: str = Field(description="Updated user preferences")


class Router(BaseModel):
    """Analyzed the unreead email and classify it according to it contents."""
    description: str = Field(description="step by step reasoning behind choosing the classification")
    classification: Literal["ignore", "notify", "respond"] = Field(
        description="The classification of the email there are only 3 possible class:" \
        "1. ignore: for irrelevant emails that does not need a response" \
        "2. notify: for important emails that doesn't need a response" \
        "3. respond: for important emails that also needs to be replied and attended to.")
    

class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]

# This state class extends the MessagesState
class TriageState(MessagesState):
    email_input: dict
    classification_decision: Literal["ignore", "respond", "notify"]


class StateInput(TypedDict):
    # This is the input to the state
    email_input: dict