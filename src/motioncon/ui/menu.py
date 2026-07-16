"""Pure navigable menu model driven by gesture events. Stdlib only."""

from __future__ import annotations

from dataclasses import dataclass

from motioncon.config import Event


@dataclass(frozen=True, slots=True)
class MenuItem:
    """One entry: a leaf with an ``action`` id, or a submenu with ``children``."""

    label: str
    action: str | None = None
    children: tuple[MenuItem, ...] = ()

    @property
    def is_submenu(self) -> bool:
        """Whether selecting this item descends into a child menu."""
        return len(self.children) > 0


@dataclass(slots=True)
class _Level:
    items: tuple[MenuItem, ...]
    index: int = 0


class Menu:
    """A list of items, a selection index, and a back-stack.

    Horizontal strokes step the selection with wrap-around. Back is unbound in
    the v1 gesture vocabulary; use ``back()`` from a keyboard fallback.
    """

    def __init__(self, items: tuple[MenuItem, ...] | list[MenuItem]) -> None:
        if not items:
            msg = "menu needs at least one item"
            raise ValueError(msg)
        self._stack: list[_Level] = [_Level(items=tuple(items))]

    @property
    def items(self) -> tuple[MenuItem, ...]:
        """Items of the current level."""
        return self._stack[-1].items

    @property
    def selected_index(self) -> int:
        """Index of the highlighted item in the current level."""
        return self._stack[-1].index

    @property
    def selected_item(self) -> MenuItem:
        """The highlighted item."""
        level = self._stack[-1]
        return level.items[level.index]

    @property
    def depth(self) -> int:
        """How many levels deep the navigation is (root = 0)."""
        return len(self._stack) - 1

    @property
    def can_go_back(self) -> bool:
        """Whether a back navigation would pop a level."""
        return len(self._stack) > 1

    def handle_event(self, event: Event) -> str | None:
        """Apply one gesture event."""
        level = self._stack[-1]
        if event is Event.STROKE_RIGHT:
            level.index = (level.index - 1) % len(level.items)
        elif event is Event.STROKE_LEFT:
            level.index = (level.index + 1) % len(level.items)
        return None

    def back(self) -> bool:
        """Pop one level off the back-stack; return whether anything popped."""
        if len(self._stack) > 1:
            self._stack.pop()
            return True
        return False
