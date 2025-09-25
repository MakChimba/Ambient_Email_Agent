from typing import Dict, List, Any, Optional
from langchain_core.tools import BaseTool

def get_tools(
    tool_names: Optional[List[str]] = None,
    *,
    include_gmail: bool = False,
    include_progress: bool = False,
) -> List[BaseTool]:
    """Get specified tools or all tools if tool_names is None.
    
    Args:
        tool_names: Optional list of tool names to include. If None, returns all tools.
        include_gmail: Whether to include Gmail tools. Defaults to False.
        include_progress: Whether to include streaming progress helper tools.
        
    Returns:
        List of tool objects
    """
    # Import default tools
    from email_assistant.tools.default.email_tools import write_email, Done, Question
    from email_assistant.tools.default.calendar_tools import (
        schedule_meeting,
        check_calendar_availability,
    )
    from email_assistant.tools.default.web_tools import google_search
    from email_assistant.tools.default.progress_tools import stream_progress
    
    # Base tools dictionary
    all_tools = {
        "write_email": write_email,
        "Done": Done,
        "Question": Question,
        "schedule_meeting": schedule_meeting,
        "check_calendar_availability": check_calendar_availability,
        "google_search": google_search,
    }

    progress_tools = {
        "stream_progress": stream_progress,
    }
    if include_progress or (
        tool_names is not None and any(name in progress_tools for name in tool_names)
    ):
        all_tools.update(progress_tools)
    
    # Add Gmail tools if requested
    if include_gmail:
        try:
            from email_assistant.tools.gmail.gmail_tools import (
                fetch_emails_tool,
                send_email_tool,
                check_calendar_tool,
                schedule_meeting_tool,
                mark_as_spam_tool,
            )

            all_tools.update({
                "fetch_emails_tool": fetch_emails_tool,
                "send_email_tool": send_email_tool,
                "check_calendar_tool": check_calendar_tool,
                "schedule_meeting_tool": schedule_meeting_tool,
                "mark_as_spam_tool": mark_as_spam_tool,
            })
        except ImportError:
            # If Gmail tools aren't available, continue without them
            pass
    
    if tool_names is None:
        return list(all_tools.values())
    
    return [all_tools[name] for name in tool_names if name in all_tools]

def get_tools_by_name(tools: Optional[List[BaseTool]] = None) -> Dict[str, BaseTool]:
    """Get a dictionary of tools mapped by name."""
    if tools is None:
        tools = get_tools()
    
    return {tool.name: tool for tool in tools}
