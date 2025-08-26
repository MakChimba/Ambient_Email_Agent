import pytest
from email_assistant.tools.gmail import gmail_tools
#!/usr/bin/env python

# Route Gmail interactions through the session's gmail_service fixture
@pytest.fixture(autouse=True)
def _route_gmail_through_fixture(gmail_service, monkeypatch):
    monkeypatch.setattr(gmail_tools, "mark_as_read", gmail_service.mark_as_read, raising=True)


import uuid
import importlib
import sys
import os
import pytest
from typing import Dict, List, Any, Tuple

from langsmith import testing as t
from langchain_core.runnables import RunnableLambda

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from email_assistant.utils import extract_tool_calls, format_messages_string

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


def load_dataset(agent_module_name: str):
    """Load the appropriate dataset for the given agent module."""

    # Force reload the base dataset module to ensure we get the latest version
    if "email_assistant.eval.email_dataset" in sys.modules:
        importlib.reload(sys.modules["email_assistant.eval.email_dataset"])

    if "gmail" in agent_module_name:
        print(f"\nINFO: Loading Gmail-specific dataset for module: {agent_module_name}")
        if "email_assistant.eval.email_gmail_dataset" in sys.modules:
            importlib.reload(sys.modules["email_assistant.eval.email_gmail_dataset"])
        from email_assistant.eval.email_gmail_dataset import (
            email_inputs,
            email_names,
            response_criteria_list,
            triage_outputs_list,
            expected_tool_calls,
        )

        # Map expected tool names to Gmail tool names
        _tool_map = {
            "write_email": "send_email_tool",
            "schedule_meeting": "schedule_meeting_tool",
            "check_calendar_availability": "check_calendar_tool",
            "done": "done",
        }
        expected_tool_calls = [
            [_tool_map.get(call.lower(), call.lower()) for call in calls]
            for calls in expected_tool_calls
        ]
        os.environ.setdefault("HITL_AUTO_ACCEPT", "1")
        # Default to live LLM for local runs; CI can override
        os.environ.setdefault("EMAIL_ASSISTANT_EVAL_MODE", "0")
        os.environ.setdefault("EMAIL_ASSISTANT_SKIP_MARK_AS_READ", "1")
        # Assume GOOGLE_API_KEY is provided via environment variables
    else:
        print(f"\nINFO: Loading standard dataset for module: {agent_module_name}")
        from email_assistant.eval.email_dataset import (
            email_inputs,
            email_names,
            response_criteria_list,
            triage_outputs_list,
            expected_tool_calls,
        )

    return (
        email_inputs,
        email_names,
        response_criteria_list,
        triage_outputs_list,
        expected_tool_calls,
    )
    
# Removed: local LLM-as-Judge setup. Use LangStudio/LangSmith UI judge for qualitative evaluation.

# Global variables for module name and imported module
AGENT_MODULE = None
agent_module = None

@pytest.fixture(autouse=True, scope="function")
def set_agent_module(agent_module_name):
    """Set the global AGENT_MODULE for each test function.
    Using scope="function" ensures we get a fresh import for each test."""
    global AGENT_MODULE, agent_module
    AGENT_MODULE = agent_module_name
    print(f"Using agent module: {AGENT_MODULE}")
    
    # Force reload the module to ensure we get the latest code
    if f"email_assistant.{AGENT_MODULE}" in sys.modules:
        importlib.reload(sys.modules[f"email_assistant.{AGENT_MODULE}"])
    
    agent_module = importlib.import_module(f"email_assistant.{AGENT_MODULE}")
    return AGENT_MODULE

def setup_assistant() -> Tuple[Any, Dict[str, Any], InMemoryStore]:
    """
    Setup the email assistant and create thread configuration.
    Returns the assistant, thread config, and store.
    """
    # Set up checkpointer and store
    checkpointer = MemorySaver()
    store = InMemoryStore()
    
    # Create a thread ID and config
    thread_id = uuid.uuid4()
    thread_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
    
    # Compile the graph based on module type
    if AGENT_MODULE in ["email_assistant_hitl_memory", "email_assistant_hitl_memory_gmail"]:
        # Memory implementation needs a store and a checkpointer
        email_assistant = agent_module.overall_workflow.compile(checkpointer=checkpointer, store=store)
    elif AGENT_MODULE in ["email_assistant_hitl"]:
        # Just use a checkpointer for HITL version
        email_assistant = agent_module.overall_workflow.compile(checkpointer=checkpointer)
    else:
        # Just use a checkpointer for other versions
        email_assistant = agent_module.overall_workflow.compile(checkpointer=checkpointer)
        store = None
    
    return email_assistant, thread_config, store

def extract_values(state: Any) -> Dict[str, Any]:
    """Extract values from state object regardless of type."""
    if hasattr(state, "values"):
        return state.values
    else:
        return state


# Helper previously used by judge evaluation removed.


# Heuristic/structured judge removed from local tests.

def run_initial_stream(email_assistant: Any, email_input: Dict, thread_config: Dict) -> List[Dict]:
    """Run the initial stream and return collected messages."""
    messages = []
    for chunk in email_assistant.stream({"email_input": email_input}, config=thread_config):
            messages.append(chunk)
    return messages

def run_stream_with_command(email_assistant: Any, command: Command, thread_config: Dict) -> List[Dict]:
    """Run stream with a command and return collected messages."""
    messages = []
    for chunk in email_assistant.stream(command, config=thread_config):
            messages.append(chunk)
    return messages

def is_module_compatible(required_modules: List[str]) -> bool:
    """Check if current module is compatible with test.
    
    Returns:
        bool: True if module is compatible, False otherwise
    """
    return AGENT_MODULE in required_modules

def create_response_test_cases(dataset: Tuple[List, List, List, List, List]):
    """Create test cases for parametrized criteria evaluation with LangSmith.
    Only includes emails that require a response (triage_output == "respond").
    These are more relevant / interesting for testing tool calling / response quality.
    """

    (
        email_inputs,
        email_names,
        response_criteria_list,
        triage_outputs_list,
        expected_tool_calls,
    ) = dataset

    test_cases = []
    for email_input, email_name, criteria, triage_output, expected_calls in zip(
        email_inputs, email_names, response_criteria_list, triage_outputs_list, expected_tool_calls
    ):
        if triage_output == "respond":
            test_cases.append((email_input, email_name, criteria, expected_calls))

    print(f"Created {len(test_cases)} test cases for emails requiring responses")
    return test_cases


def pytest_generate_tests(metafunc):
    """Parametrize tests dynamically based on the selected agent module."""

    required = {"email_input", "email_name", "criteria", "expected_calls"}
    if required.issubset(set(metafunc.fixturenames)):
        agent_module_name = metafunc.config.getoption("--agent-module")
        dataset = load_dataset(agent_module_name)
        test_cases = create_response_test_cases(dataset)
        metafunc.parametrize(
            "email_input,email_name,criteria,expected_calls",
            test_cases,
        )

# Reference output key
def test_email_dataset_tool_calls(email_input, email_name, criteria, expected_calls, gmail_service):
    """Test if email processing contains expected tool calls."""
    # Log minimal inputs for LangSmith (safe noop if plugin disabled)
    try:
        t.log_inputs({"module": AGENT_MODULE, "test": "test_email_dataset_tool_calls"})
    except Exception:
        pass
    
    print(f"Processing {email_name}...")
    
    # Set up the assistant
    email_assistant, thread_config, _ = setup_assistant()
    
    # Run the agent        
    if AGENT_MODULE in ["email_assistant", "email_assistant_hitl_memory_gmail"]:
        # Workflow agent takes email_input directly
        result = email_assistant.invoke({"email_input": email_input}, config=thread_config)
    else:
        raise ValueError(f"Unsupported agent module: {AGENT_MODULE}. Only 'email_assistant' is supported in automated testing.")
        
    # Get the final state
    state = email_assistant.get_state(thread_config)
    values = extract_values(state)
        
    # Extract tool calls from messages
    extracted_tool_calls = extract_tool_calls(values["messages"])
            
    # Check if all expected tool calls are in the extracted ones
    missing_calls = [call for call in expected_calls if call.lower() not in extracted_tool_calls]
    # Extra calls are allowed (we only fail if expected calls are missing)
    extra_calls = [call for call in extracted_tool_calls if call.lower() not in [c.lower() for c in expected_calls]]
   
    # Log 
    all_messages_str = format_messages_string(values["messages"])
    try:
        t.log_outputs({
                    "extracted_tool_calls": extracted_tool_calls,
                    "missing_calls": missing_calls,
                    "extra_calls": extra_calls,
                    "response": all_messages_str
                })
    except Exception:
        pass

    # Pass feedback key
    assert len(missing_calls) == 0
            
# Reference output key
# Each test case is (email_input, email_name, criteria, expected_calls)
# Qualitative judge-based test removed. Use the LangStudio UI judge instead.
 
