from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from .types import JsonDict, NormalizedResponse


def default_projector(response: NormalizedResponse) -> JsonDict:
    """Default JSONL record shape: id, model, output text, usage, cost, error."""
    return {
        "id": response.id,
        "model": response.model,
        "provider": response.provider,
        "output_text": response.output_text,
        "usage": response.usage.to_dict(),
        "billing": response.billing,
        "cost": response.cost.to_dict() if response.cost is not None else None,
        "error": response.error,
    }


class ResultSink(ABC):
    """Destination for batch results; also provides done-ids for resume."""

    @abstractmethod
    def save(self, response: NormalizedResponse) -> None:
        """Persist a single response. Must be safe to call from multiple threads."""
        raise NotImplementedError

    @abstractmethod
    def load_done_ids(self) -> set[str]:
        """Return the ids already persisted, used to skip work on re-runs."""
        raise NotImplementedError


class PerRequestJsonSink(ResultSink):
    """Writes one ``<id>.json`` file per response into ``output_dir``.

    Stores the raw provider response when available, otherwise the normalized
    response dict.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def save(self, response: NormalizedResponse) -> None:
        path = self.output_dir / f"{response.id}.json"
        payload = response.raw_response if response.raw_response is not None else response.to_dict()
        with self._lock:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_done_ids(self) -> set[str]:
        return {path.stem for path in self.output_dir.glob("*.json")}


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
        record = self.projector(response)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

    def load_done_ids(self) -> set[str]:
        done: set[str] = set()
        if not self.path.exists():
            return done
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                item_id = payload.get("id")
                if item_id is not None:
                    done.add(str(item_id))
        return done
