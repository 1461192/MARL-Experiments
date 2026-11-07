"""Tiny name -> class registries so new agents / games can be plugged in
without touching the runner or the CLI."""
from __future__ import annotations

from typing import Callable, Dict


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, type] = {}

    def register(self, name: str) -> Callable[[type], type]:
        def deco(cls: type) -> type:
            self._items[name] = cls          # re-registering replaces (handy in notebooks)
            cls.registry_name = name
            return cls
        return deco

    def get(self, name: str) -> type:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {self.names()}") from None

    def create(self, name: str, **kwargs):
        return self.get(name)(**kwargs)

    def names(self):
        return sorted(self._items)


AGENTS = Registry("agent")
ENVS = Registry("env")

register_agent = AGENTS.register
register_env = ENVS.register
