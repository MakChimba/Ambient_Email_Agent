# TODO: S3 - Implement graph nodes for creating, canceling, and checking reminders.


def create_reminder_node(state):
    """Graph node to create a reminder based on triage decision."""
    print("INFO: [Node] create_reminder_node (not implemented)")
    # This node will check the triage result and call ReminderStore.add_reminder
    return state


def cancel_reminder_node(state):
    """Graph node to cancel a reminder if a reply is detected."""
    print("INFO: [Node] cancel_reminder_node (not implemented)")
    # This node will check for user replies and call ReminderStore.cancel_reminder
    return state

