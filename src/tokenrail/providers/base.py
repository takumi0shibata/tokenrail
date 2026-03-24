from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from ..types import NormalizedResponse


class BaseProvider(ABC):
    name = "base"
    supports_batching = False
    batch_size = 1

    @abstractmethod
    def create(self, **kwargs: Any) -> NormalizedResponse:
        raise NotImplementedError

    def create_many(self, requests: Sequence[dict[str, Any]]) -> list[NormalizedResponse]:
        return [self.create(**request) for request in requests]
