from IPython.display import Image, display
from typing import Literal
from langgraph.graph import StateGraph, END, START
from gmail_utils.response import interrupt_handler, llm_call, should_continue, mark_as_read_node
from gmail_utils.schemas import StateInput, State, TriageState
from gmail_utils.triage import triage_interrupt_handler, triage_router


# Build workflow
agent_builder = StateGraph(State)

# Add nodes - with store parameter
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("interrupt_handler", interrupt_handler)
agent_builder.add_node("mark_as_read_node", mark_as_read_node)

# Add edges
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "interrupt_handler": "interrupt_handler",
        "mark_as_read_node":"mark_as_read_node",
    },
)
agent_builder.add_edge("mark_as_read_node", END)

# Compile the agent
response_agent = agent_builder.compile()

# Build overall workflow with store and checkpointer
overall_workflow = (
    StateGraph(TriageState, input_schema=StateInput)
    .add_node(triage_router)
    .add_node(triage_interrupt_handler)
    .add_node("response_agent", response_agent)
    .add_node("mark_as_read_node", mark_as_read_node)
    .add_edge(START, "triage_router")
    .add_edge("mark_as_read_node", END)
)

email_assistant = overall_workflow.compile()

if __name__ == "__main__":
    import os
    file_name = "mermaid_graph.png"
    image_bytes = email_assistant.get_graph(xray=True).draw_mermaid_png()
    output_path = os.path.join(os.getcwd(), file_name)
    with open(output_path, "wb") as f:
        f.write(image_bytes)
    display(Image(email_assistant.get_graph(xray=True).draw_mermaid_png()))