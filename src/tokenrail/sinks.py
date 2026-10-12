from __future__ import annotations

import json
import os
import tempfile
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from .types import JsonDict, NormalizedResponse, to_jsonable


def _is_done_record(payload: object) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("id") is not None
        and payload.get("error") is None
        and payload.get("status") in (None, "completed")
    )


def default_projector(response: NormalizedResponse) -> JsonDict:
    """Default JSONL record shape: id, model, output text, usage, cost, error."""
    return {
        "id": response.id,
        "model": response.model,
        "provider": response.provider,
        "output_text": response.output_text,
        "output_parsed": to_jsonable(response.output_parsed),
        "refusal": response.refusal,
        "usage": response.usage.to_dict(),
        "billing": response.billing,
        "cost": response.cost.to_dict() if response.cost is not None else None,
        "error": response.error,
        "status": response.status,
    }


class ResultSink(ABC):
    """Destination for batch results; also provides done-ids for resume."""

    def validate_id(self, item_id: str) -> None:
        """Validate an id before sending a request; by default any id is allowed."""
        return None

    @abstractmethod
    def save(self, response: NormalizedResponse) -> None:
        """Persist a single response. Must be safe to call from multiple threads."""
        raise NotImplementedError

    @abstractmethod
    def load_done_ids(self) -> set[str]:
        """Return ids with valid, successfully completed records for resume."""
        raise NotImplementedError


class PerRequestJsonSink(ResultSink):
    """Writes one ``<id>.json`` file per response into ``output_dir``.

    Stores the raw provider response for successful requests when available.
    Errors use the normalized response dict, including the raw response when
    available, so local parsing failures remain distinguishable on resume.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def save(self, response: NormalizedResponse) -> None:
        self.validate_id(response.id)
        path = self.output_dir / f"{response.id}.json"
        payload = (
            response.raw_response
            if response.raw_response is not None and response.error is None
            else response.to_dict()
        )
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        with self._lock:
            # Replace the directory entry, never follow an existing destination
            # symlink. Failed writes leave the previous complete file intact.
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self.output_dir, prefix=".tokenrail-", delete=False
                ) as handle:
                    temporary_path = Path(handle.name)
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    def validate_id(self, item_id: str) -> None:
        forbidden = '<>:"/\\|?*\x00'
        reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
        if (
            not item_id
            or item_id in {".", ".."}
            or any(char in forbidden or ord(char) < 32 for char in item_id)
            or item_id.endswith((".", " "))
            or item_id.split(".", 1)[0].upper() in reserved
        ):
            raise ValueError(f"Unsafe per-request file id: {item_id!r}")

    def load_done_ids(self) -> set[str]:
        done: set[str] = set()
        with self._lock:
            for path in self.output_dir.glob("*.json"):
                try:
                    self.validate_id(path.stem)
                    if path.is_symlink():
                        continue
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    continue
                if _is_done_record(payload):
                    done.add(path.stem)
        return done


class ResultsJsonlSink(ResultSink):
    """Appends one JSON line per response to ``path``.

    ``projector`` maps a :class:`~tokenrail.types.NormalizedResponse` to the
    record to write; it must include an ``"id"`` key for resume to work.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        projector: Callable[[NormalizedResponse], JsonDict] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.projector = projector or default_projector
        self._lock = threading.Lock()

    def save(self, response: NormalizedResponse) -> None:
        # Completion metadata is authoritative even when a custom projector
        # omits it or accidentally labels an unsuccessful result as complete.
        record = {**self.projector(response), "error": response.error, "status": response.status}
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
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())

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
