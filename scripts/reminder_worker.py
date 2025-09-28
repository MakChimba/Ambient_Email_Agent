import argparse
import time
import os
import sys
from dotenv import load_dotenv

# Add src to path to allow for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from email_assistant.tools.reminders import (
    ReminderStore,
    get_default_store,
    get_default_delivery,
)


def check_reminders(store: ReminderStore):
    """Core logic to check for and process due reminders."""
    print("INFO: Running reminder check...")
    delivery = get_default_delivery()
    try:
        due_reminders = store.get_due_reminders()
        if not due_reminders:
            print("INFO: No due reminders found.")
            return

        print(f"INFO: Found {len(due_reminders)} due reminder(s).")
        for reminder in due_reminders:
            try:
                print(f"- Processing reminder {reminder.id} for thread {reminder.thread_id}")
                delivery.send_notification(reminder)
                store.mark_as_notified(reminder.id)
                print(f"  - Marked as notified.")
            except Exception as e:
                print(f"ERROR: Failed to process reminder {reminder.id}: {e}")
    except Exception as e:
        print(f"ERROR: Could not check reminders: {e}")
    print("INFO: Reminder check complete.")

def list_reminders(store: ReminderStore):
    """Lists all active (pending) reminders."""
    print("--- Active Reminders ---")
    reminders = list(store.iter_active_reminders())

    if not reminders:
        print("No active reminders found.")
        return

    for reminder in reminders:
        due = reminder.due_at
        due_display = due.isoformat() if hasattr(due, "isoformat") else str(due)
        print(
            f"- ID: {reminder.id}\n"
            f"  Thread: {reminder.thread_id}\n"
            f"  Subject: {reminder.subject}\n"
            f"  Due: {due_display}"
        )

def cancel_reminder_cli(store: ReminderStore, thread_id: str):
    """CLI command to cancel a reminder."""
    print(f"Attempting to cancel reminders for thread: {thread_id}...")
    cancelled_count = store.cancel_reminder(thread_id)
    if cancelled_count > 0:
        print(f"✅ Successfully cancelled {cancelled_count} reminder(s).")
    else:
        print("No active reminder found for that thread ID.")

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Reminder worker and admin tool for the Email Assistant.")
    parser.add_argument("--once", action="store_true", help="Run the check for due reminders once and exit.")
    parser.add_argument("--loop", action="store_true", help="Run the check in a continuous loop.")
    parser.add_argument("--list", action="store_true", help="List all active reminders.")
    parser.add_argument("--cancel", type=str, metavar="THREAD_ID", help="Cancel an active reminder by its thread ID.")
    args = parser.parse_args()

    store = get_default_store()

    if args.loop:
        poll_interval_min = int(os.getenv("REMINDER_POLL_INTERVAL_MIN", 15))
        print(f"Starting reminder worker in loop mode (checking every {poll_interval_min} minutes). Press Ctrl+C to exit.")
        while True:
            try:
                check_reminders(store)
                print(f"INFO: Sleeping for {poll_interval_min} minutes...")
                time.sleep(poll_interval_min * 60)
            except KeyboardInterrupt:
                print("\nCaught Ctrl+C. Shutting down worker.")
                break
    elif args.once:
        check_reminders(store)
    elif args.list:
        list_reminders(store)
    elif args.cancel:
        cancel_reminder_cli(store, args.cancel)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
