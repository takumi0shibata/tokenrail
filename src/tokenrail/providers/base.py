from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import NormalizedResponse


class BaseProvider(ABC):
    """Interface for backends that execute a single request.

    Implementations accept Responses-API-style keyword arguments and return a
    :class:`~tokenrail.types.NormalizedResponse`.
    """

    name = "base"

    @abstractmethod
    def create(self, **kwargs: Any) -> NormalizedResponse:
        raise NotImplementedError

    def parse(self, **kwargs: Any) -> NormalizedResponse:
        raise NotImplementedError(f"{self.name} does not support structured output parsing")
