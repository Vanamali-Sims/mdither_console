"""Tests for the menu's v1 gesture vocabulary."""

from __future__ import annotations

import pytest

from motioncon.config import Event
from motioncon.ui.menu import Menu, MenuItem


def demo_menu() -> Menu:
    return Menu(
        (
            MenuItem("Play", action="play"),
            MenuItem("Gallery", action="gallery"),
            MenuItem("About", action="about"),
        )
    )


def test_stroke_left_steps_forward_and_wraps() -> None:
    menu = demo_menu()
    for _ in range(3):
        menu.handle_event(Event.STROKE_LEFT)
    assert menu.selected_index == 0


def test_stroke_right_steps_backward_and_wraps() -> None:
    menu = demo_menu()
    menu.handle_event(Event.STROKE_RIGHT)
    assert menu.selected_index == 2


def test_gestures_do_not_navigate_back() -> None:
    menu = demo_menu()
    menu.handle_event(Event.STROKE_LEFT)
    menu.handle_event(Event.STROKE_RIGHT)
    assert menu.depth == 0
    assert not menu.can_go_back
    assert not menu.back()


def test_initial_state() -> None:
    menu = demo_menu()
    assert menu.selected_item.label == "Play"
    assert menu.depth == 0
    assert not menu.can_go_back


def test_empty_menu_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Menu(())


def test_is_submenu_flag() -> None:
    leaf = MenuItem("Leaf", action="x")
    parent = MenuItem("Parent", children=(leaf,))
    assert parent.is_submenu
    assert not leaf.is_submenu
