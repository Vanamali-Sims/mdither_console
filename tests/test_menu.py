"""Tests for the pure menu model."""

from __future__ import annotations

import pytest

from motioncon.config import Event
from motioncon.ui.menu import Menu, MenuItem


def demo_menu() -> Menu:
    return Menu(
        (
            MenuItem("Play", action="play"),
            MenuItem(
                "Gallery",
                children=(
                    MenuItem("Photos", action="gallery.photos"),
                    MenuItem("Videos", action="gallery.videos"),
                ),
            ),
            MenuItem("About", action="about"),
        )
    )


class TestNavigation:
    def test_initial_state(self) -> None:
        menu = demo_menu()
        assert menu.selected_index == 0
        assert menu.selected_item.label == "Play"
        assert menu.depth == 0
        assert not menu.can_go_back

    def test_down_and_right_step_forward(self) -> None:
        menu = demo_menu()
        menu.handle_event(Event.SWIPE_DOWN)
        assert menu.selected_index == 1
        menu.handle_event(Event.SWIPE_RIGHT)
        assert menu.selected_index == 2

    def test_up_and_left_step_backward_with_wrap(self) -> None:
        menu = demo_menu()
        menu.handle_event(Event.SWIPE_UP)
        assert menu.selected_index == 2  # wrapped
        menu.handle_event(Event.SWIPE_LEFT)
        assert menu.selected_index == 1

    def test_forward_wraps_at_end(self) -> None:
        menu = demo_menu()
        for _ in range(3):
            menu.handle_event(Event.SWIPE_DOWN)
        assert menu.selected_index == 0


class TestSelect:
    def test_select_leaf_returns_action(self) -> None:
        menu = demo_menu()
        assert menu.handle_event(Event.SELECT) == "play"
        assert menu.depth == 0  # activating a leaf does not navigate

    def test_select_submenu_descends(self) -> None:
        menu = demo_menu()
        menu.handle_event(Event.SWIPE_DOWN)  # Gallery
        assert menu.handle_event(Event.SELECT) is None
        assert menu.depth == 1
        assert menu.selected_index == 0
        assert [item.label for item in menu.items] == ["Photos", "Videos"]

    def test_leaf_without_action_returns_label(self) -> None:
        menu = Menu((MenuItem("Bare"),))
        assert menu.handle_event(Event.SELECT) == "Bare"


class TestBack:
    def test_double_swipe_left_pops_level(self) -> None:
        menu = demo_menu()
        menu.handle_event(Event.SWIPE_DOWN)
        menu.handle_event(Event.SELECT)
        assert menu.can_go_back
        menu.handle_event(Event.DOUBLE_SWIPE_LEFT)
        assert menu.depth == 0
        assert menu.selected_item.label == "Gallery"  # selection preserved

    def test_back_at_root_is_noop(self) -> None:
        menu = demo_menu()
        menu.handle_event(Event.DOUBLE_SWIPE_LEFT)
        assert menu.depth == 0
        assert not menu.back()

    def test_selection_inside_submenu_independent(self) -> None:
        menu = demo_menu()
        menu.handle_event(Event.SWIPE_DOWN)
        menu.handle_event(Event.SELECT)
        menu.handle_event(Event.SWIPE_DOWN)
        assert menu.selected_item.label == "Videos"
        menu.back()
        assert menu.selected_item.label == "Gallery"


class TestConstruction:
    def test_empty_menu_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            Menu(())

    def test_is_submenu_flag(self) -> None:
        leaf = MenuItem("Leaf", action="x")
        parent = MenuItem("Parent", children=(leaf,))
        assert parent.is_submenu
        assert not leaf.is_submenu
