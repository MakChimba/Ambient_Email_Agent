"""Reminder storage and delivery (S2 implementation).

This module provides:
- A `Reminder` dataclass for reminder records
- An abstract `ReminderStore` with a SQLite-backed implementation `SqliteReminderStore`
- An abstract `ReminderDelivery` with a simple console notifier and a Gmail placeholder

Environment variables used:
- `REMINDER_DB_PATH` (default: `.local/reminders.db`)
"""

from __future__ import annotations

import abc
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence


# ------------------------
# Data model
# ------------------------


@dataclass
class Reminder:
    id: str
    thread_id: str
    subject: str
    due_at: datetime
    reason: str
    status: str
    created_at: datetime
    notified_at: Optional[datetime] = None
    canceled_at: Optional[datetime] = None


# ------------------------
# Storage interface
# ------------------------


class ReminderStore(abc.ABC):
    """Interface for storing and managing reminders."""

    @abc.abstractmethod
    def setup(self) -> None:
        """Initialize the database and create tables if they don't exist."""
        raise NotImplementedError

    @abc.abstractmethod
    def add_reminder(self, thread_id: str, subject: str, due_at: datetime, reason: str) -> str:
        """Add a new reminder, returning its ID."""
        raise NotImplementedError

    @abc.abstractmethod
    def cancel_reminder(self, thread_id: str) -> int:
        """Cancel active reminders for a given thread. Returns count affected."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_due_reminders(self) -> List[Reminder]:
        """Get reminders past their `due_at` that are not canceled/notified."""
        raise NotImplementedError

    @abc.abstractmethod
    def mark_as_notified(self, reminder_id: str) -> None:
        """Mark a reminder as notified."""
        raise NotImplementedError

    @abc.abstractmethod
    def iter_active_reminders(self) -> List[Reminder]:
        """Return active reminders that have not been cancelled or notified."""
        raise NotImplementedError

    @abc.abstractmethod
    def apply_actions(self, actions: Sequence[Dict[str, object]]) -> Dict[str, object]:
        """Apply a batch of reminder actions atomically.

        The dispatcher provides actions shaped as dictionaries with at least an
        ``action`` key (``"cancel"`` or ``"create"``) and a ``thread_id``.  The
        implementation must cancel existing reminders before creating new ones in
        the same transaction so the workflow never observes a cancelled reminder
        without the corresponding recreation.
        """
        raise NotImplementedError


# ------------------------
# SQLite implementation
# ------------------------


class SqliteReminderStore(ReminderStore):
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("REMINDER_DB_PATH", ".local/reminders.db")
        self._connection: Optional[sqlite3.Connection] = None
        # Ensure directory exists for file-based dbs
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """Returns a new or cached database connection."""
        if self.db_path == ":memory:":
            if self._connection is None:
                self._connection = sqlite3.connect(":memory:", check_same_thread=False)
                self._connection.row_factory = sqlite3.Row
            return self._connection
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def setup(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                  id TEXT PRIMARY KEY,
                  thread_id TEXT NOT NULL,
                  subject TEXT NOT NULL,
                  due_at TEXT NOT NULL,
                  reason TEXT NOT NULL,
                  status TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  notified_at TEXT,
                  canceled_at TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reminders_thread ON reminders(thread_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(due_at)")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_active_reminder ON reminders(thread_id) WHERE canceled_at IS NULL AND notified_at IS NULL")

    def add_reminder(
        self,
        thread_id: str,
        subject: str,
        due_at: datetime,
        reason: str,
    ) -> str:
        """Adds a reminder if no active one exists for the thread_id. Idempotent."""
        self.setup()
        with self._connect() as conn:
            return self._add_reminder(conn, thread_id, subject, due_at, reason)

    def get_active_reminder_for_thread(self, thread_id: str) -> Optional[Reminder]:
        """Fetches the active (not canceled, not notified) reminder for a thread."""
        self.setup()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM reminders WHERE thread_id = ? AND canceled_at IS NULL AND notified_at IS NULL",
                (thread_id,),
            ).fetchone()
        return self._row_to_reminder(row) if row else None

    def cancel_reminder(self, thread_id: str) -> int:
        self.setup()
        with self._connect() as conn:
            return self._cancel_reminder(conn, thread_id)

    def get_due_reminders(self) -> List[Reminder]:
        now_iso = self._now_iso()
        self.setup()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reminders WHERE status = 'pending' AND due_at <= ?",
                (now_iso,),
            ).fetchall()
        return [self._row_to_reminder(r) for r in rows]

    def mark_as_notified(self, reminder_id: str) -> None:
        self.setup()
        with self._connect() as conn:
            conn.execute(
                "UPDATE reminders SET status = 'notified', notified_at = ? WHERE id = ?",
                (self._now_iso(), reminder_id),
            )

    def iter_active_reminders(self) -> List[Reminder]:
        self.setup()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reminders WHERE status = 'pending' AND canceled_at IS NULL AND notified_at IS NULL ORDER BY due_at ASC",
            ).fetchall()
        return [self._row_to_reminder(r) for r in rows]

    def apply_actions(self, actions: Sequence[Dict[str, object]]) -> Dict[str, object]:
        """Apply reminder actions in a single transaction, cancelling before creating."""

        self.setup()
        if not actions:
            return {"cancelled": {}, "created": {}}

        cancelled: Dict[str, int] = {}
        created: Dict[str, str] = {}

        with self._connect() as conn:
            try:
                # Cancels first to avoid duplicate reminders when recreating
                for action in actions:
                    if not isinstance(action, dict):
                        continue
                    if str(action.get("action", "")).lower() != "cancel":
                        continue
                    thread_id = str(action.get("thread_id") or "").strip()
                    if not thread_id:
                        continue
                    count = self._cancel_reminder(conn, thread_id)
                    if count:
                        cancelled[thread_id] = cancelled.get(thread_id, 0) + count

                for action in actions:
                    if not isinstance(action, dict):
                        continue
                    if str(action.get("action", "")).lower() != "create":
                        continue
                    thread_id = str(action.get("thread_id") or "").strip()
                    if not thread_id:
                        continue
                    subject = str(action.get("subject") or "(no subject)")
                    due_at = self._ensure_datetime(action.get("due_at"))
                    reason = str(action.get("reason") or "Reminder created via dispatcher")
                    reminder_id = self._add_reminder(conn, thread_id, subject, due_at, reason)
                    if reminder_id:
                        created[thread_id] = reminder_id
            except Exception:
                conn.rollback()
                raise

        return {"cancelled": cancelled, "created": created}

    # -------- helpers --------
    @staticmethod
    def _to_iso(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat()

    @staticmethod
    def _parse_iso(s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _row_to_reminder(self, row: sqlite3.Row) -> Reminder:
        return Reminder(
            id=row["id"],
            thread_id=row["thread_id"],
            subject=row["subject"],
            due_at=self._parse_iso(row["due_at"]) or datetime.now(timezone.utc),
            reason=row["reason"],
            status=row["status"],
            created_at=self._parse_iso(row["created_at"]) or datetime.now(timezone.utc),
            notified_at=self._parse_iso(row["notified_at"]) if row["notified_at"] else None,
            canceled_at=self._parse_iso(row["canceled_at"]) if row["canceled_at"] else None,
        )

    def _ensure_datetime(self, value: object) -> datetime:
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str):
            val = value.strip()
            if not val:
                dt = datetime.now(timezone.utc)
            else:
                if val.endswith("Z"):
                    val = val[:-1] + "+00:00"
                try:
                    dt = datetime.fromisoformat(val)
                except ValueError:
                    dt = datetime.now(timezone.utc)
        else:
            dt = datetime.now(timezone.utc)

        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _cancel_reminder(self, conn: sqlite3.Connection, thread_id: str) -> int:
        cur = conn.execute(
            "UPDATE reminders SET status = 'canceled', canceled_at = ? WHERE thread_id = ? AND canceled_at IS NULL AND notified_at IS NULL",
            (self._now_iso(), thread_id),
        )
        return cur.rowcount or 0

    def _add_reminder(
        self,
        conn: sqlite3.Connection,
        thread_id: str,
        subject: str,
        due_at: datetime,
        reason: str,
    ) -> str:
        existing = conn.execute(
            "SELECT id FROM reminders WHERE thread_id = ? AND canceled_at IS NULL AND notified_at IS NULL",
            (thread_id,),
        ).fetchone()
        if existing:
            print(f"INFO: Active reminder already exists for thread {thread_id}.")
            return existing["id"]
        reminder_id = str(uuid.uuid4())
        try:
            conn.execute(
                "INSERT INTO reminders (id, thread_id, subject, due_at, reason, status, created_at) VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                (reminder_id, thread_id, subject, self._to_iso(due_at), reason, self._now_iso()),
            )
        except sqlite3.IntegrityError:
            print(f"INFO: Race condition prevented duplicate reminder for thread {thread_id}.")
            existing = conn.execute(
                "SELECT id FROM reminders WHERE thread_id = ? AND canceled_at IS NULL AND notified_at IS NULL",
                (thread_id,),
            ).fetchone()
            return existing["id"] if existing else ""
        return reminder_id


# ------------------------
# Delivery interface + stubs
# ------------------------


class ReminderDelivery(abc.ABC):
    """Interface for delivering reminder notifications."""

    @abc.abstractmethod
    def send_notification(self, reminder: Reminder) -> None:
        """Send a notification for a due reminder."""
        raise NotImplementedError


class ConsoleReminderDelivery(ReminderDelivery):
    """Simple console-based notifier (useful for local/dev)."""

    def send_notification(self, reminder: Reminder) -> None:
        print(
            f"REMINDER DUE:\n- id: {reminder.id}\n- thread_id: {reminder.thread_id}\n- subject: {reminder.subject}\n- due_at: {reminder.due_at.isoformat()}\n- reason: {reminder.reason}"
        )


class GmailNotifier(ReminderDelivery):
    """Placeholder for Gmail-based delivery."""

    def send_notification(self, reminder: Reminder) -> None:
        print(
            f"[GMAIL PLACEHOLDER] Would send reminder for thread '{reminder.thread_id}' to '{os.getenv('REMINDER_NOTIFY_EMAIL', 'me@example.com')}': {reminder.subject}"
        )


# ------------------------
# Factory helpers
# ------------------------


def get_default_store() -> SqliteReminderStore:
    store = SqliteReminderStore()
    store.setup()
    return store


def get_default_delivery() -> ReminderDelivery:
    return ConsoleReminderDelivery()
