"""
Append-only Audit Logger for tracking research-demo interactions.
Records inference timestamps, event types, latency, and status WITHOUT storing raw image bytes or PHI.
NOTE: This is a demonstration audit log and is NOT claimed to be tamper-proof or HIPAA/GDPR certified.
"""
from pathlib import Path
import json
import time
import threading
from typing import Dict, Any, List, Optional

_LOCK = threading.Lock()


class AuditLogger:
    """
    Thread-safe, append-only audit logger writing to a JSONL log file.
    """

    def __init__(self, log_path: Optional[Path | str] = None):
        if log_path is None:
            # Place in artifacts directory
            root = Path(__file__).resolve().parent.parent.parent
            self.log_path = root / "artifacts" / "audit_events.jsonl"
        else:
            self.log_path = Path(log_path)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        status: str = "SUCCESS",
    ) -> Dict[str, Any]:
        """
        Appends an event record to the audit file.
        details must NOT contain raw image bytes or PHI.
        """
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "epoch_time": round(time.time(), 3),
            "event_type": event_type,
            "status": status,
            "details": details,
        }

        # Thread-safe write
        with _LOCK:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        return record

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Reads the most recent audit records from disk."""
        if not self.log_path.is_file():
            return []

        records = []
        with _LOCK:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            continue
        return records[-limit:]


_audit_singleton: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Singleton getter for AuditLogger."""
    global _audit_singleton
    if _audit_singleton is None:
        _audit_singleton = AuditLogger()
    return _audit_singleton
