from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import NormalizedResponse


class BaseProvider(ABC):
    name = "base"

    @abstractmethod
    def create(self, **kwargs: Any) -> NormalizedResponse:
        raise NotImplementedError
