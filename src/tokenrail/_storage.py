from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from pathlib import Path

from .types import JsonDict, NormalizedResponse


def _is_done_record(payload: object) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("id") is not None
        and payload.get("error") is None
        and payload.get("status") in (None, "completed")
    )


class _JsonlWriter:
    """Internal append-only result storage with successful-id lookup."""

    def __init__(
        self,
        path: str | Path,
        *,
        projector: Callable[[NormalizedResponse], JsonDict] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.projector = projector if projector is not None else NormalizedResponse.to_dict
        self._lock = threading.Lock()

    def save(self, response: NormalizedResponse) -> None:
        # Completion metadata is authoritative even when a custom projector
        # omits it or accidentally labels an unsuccessful result as complete.
        record = {**self.projector(response), "id": response.id, "error": response.error, "status": response.status}
        serialized = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock:
            with self.path.open("a+b") as handle:
                # A crash may leave a partial final line. Keep complete records
                # (including one without a newline), discard only a broken tail.
                handle.seek(0, os.SEEK_END)
                end = handle.tell()
                if end:
                    handle.seek(end - 1)
                    if handle.read(1) != b"\n":
                        start = end
                        while start:
                            chunk_start = max(0, start - 8192)
                            handle.seek(chunk_start)
                            chunk = handle.read(start - chunk_start)
                            newline = chunk.rfind(b"\n")
                            if newline >= 0:
                                start = chunk_start + newline + 1
                                break
                            start = chunk_start
                        handle.seek(start)
                        tail = handle.read(end - start)
                        try:
                            payload = json.loads(tail)
                            valid = isinstance(payload, dict)
                        except (ValueError, UnicodeError):
                            valid = False
                        if valid:
                            handle.write(b"\n")
                        else:
                            handle.truncate(start)
                handle.seek(0, os.SEEK_END)
                committed_end = handle.tell()
                try:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                except Exception:
                    # Do not let a failed append become a successful resume id.
                    handle.seek(committed_end)
                    handle.truncate()
                    handle.flush()
                    raise

    def load_done_ids(self) -> set[str]:
        done: set[str] = set()
        if not self.path.exists():
            return done
        with self._lock, self.path.open("rb") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if _is_done_record(payload):
                    done.add(str(payload["id"]))
        return done
