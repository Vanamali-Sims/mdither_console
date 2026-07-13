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

    Swipes step the selection (up/left = previous, down/right = next, with
    wrap-around), SELECT descends into submenus or activates leaves, and
    DOUBLE_SWIPE_LEFT pops back one level.
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
        """Apply one gesture event; return the activated action id, if any."""
        level = self._stack[-1]
        if event in (Event.SWIPE_UP, Event.SWIPE_LEFT):
            level.index = (level.index - 1) % len(level.items)
        elif event in (Event.SWIPE_DOWN, Event.SWIPE_RIGHT):
            level.index = (level.index + 1) % len(level.items)
        elif event is Event.DOUBLE_SWIPE_LEFT:
            self.back()
        elif event is Event.SELECT:
            item = self.selected_item
            if item.is_submenu:
                self._stack.append(_Level(items=item.children))
            else:
                return item.action if item.action is not None else item.label
        return None

    def back(self) -> bool:
        """Pop one level off the back-stack; return whether anything popped."""
        if len(self._stack) > 1:
            self._stack.pop()
            return True
        return False
