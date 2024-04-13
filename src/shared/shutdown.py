from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ShutdownState:
    is_shutdown_requested: bool = field(init=False, default=False)


__all__ = ['ShutdownState']
