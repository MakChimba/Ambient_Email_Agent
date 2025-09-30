# Agent Diagrams

This reference collects Mermaid diagrams for each LangGraph workflow included in the project. All graphs use sync durability checkpoints and share the same reminder dispatcher primitives; arrows focus on high-level routing.

## `email_assistant`
```mermaid
flowchart TD
    START --> TRIAGE[triage_router]
    TRIAGE --> RESPONSE[response_agent]
    RESPONSE --> MARK[mark_as_read_node]
    MARK --> END
```

## `email_assistant_hitl`
```mermaid
flowchart TD
    START --> TRIAGE[triage_router]
    TRIAGE -->|respond| RESPONSE[response_agent]
    TRIAGE -->|notify| HITL[triage_interrupt_handler]
    HITL -->|respond| RESPONSE
    HITL -->|ignore / accept| END
    RESPONSE --> MARK[mark_as_read_node]
    MARK --> END
```

## `email_assistant_hitl_memory`
```mermaid
flowchart TD
    START --> TRIAGE[triage_router]
    TRIAGE --> MEMORY[update_memory]
    MEMORY --> APPLY[apply_reminder_actions_node]
    TRIAGE -->|notify| HITL[triage_interrupt_handler]
    APPLY --> RESPONSE[response_agent]
    RESPONSE --> MARK[mark_as_read_node]
    MARK --> END
    HITL -->|respond| APPLY
    HITL -->|ignore / accept| END
```

## `email_assistant_hitl_memory_gmail`
```mermaid
flowchart TD
    START --> TRIAGE[triage_router]
    TRIAGE --> APPLY[apply_reminder_actions_node]
    TRIAGE -->|notify/HITL| HITL[triage_interrupt_handler]
    APPLY --> RESPONSE[response_agent]
    RESPONSE --> MARK[mark_as_read_node]
    MARK --> END
    HITL -->|respond| APPLY
    HITL -->|ignore / accept| END
```

