"""The Player's state logic, exercised without opening a window.

Only `start()` and `_draw()` touch the display; everything asserted here is
pure bookkeeping, so these run headless.
"""

import pygame
import pytest

from cookie_env.envs.goal_grid import GoalGrid
from cookie_env.envs.lava_grid import LavaGrid
from cookie_env.utils.play import Player

FORWARD = 2
GOAL = (7, 7)


def make_player(cls=LavaGrid, **kwargs):
    kwargs.setdefault("size", 9)
    kwargs.setdefault("goal_pos", GOAL)
    kwargs.setdefault("agent_start_pos", (1, 1))
    if cls is not GoalGrid:
        kwargs.setdefault("n_lava", 1)
    player = Player(cls(render_mode="rgb_array", **kwargs))
    player._reset()
    return player


def walk_onto(player, cell):
    x, y = cell
    player.core.agent_pos = (x - 1, y)
    player.core.agent_dir = 0
    player._step(FORWARD)


class TestBookkeeping:
    def test_return_accumulates(self):
        p = make_player()
        walk_onto(p, GOAL)
        assert p.total == 1.0
        assert p.last_reward == 1.0

    def test_paid_once_shows_in_the_return(self):
        """The HUD must not suggest the bonus is collectable twice."""
        p = make_player()
        walk_onto(p, GOAL)
        p._step(0)
        assert p.last_reward == 0.0
        assert p.total == 1.0

    def test_reset_clears_the_episode(self):
        p = make_player()
        walk_onto(p, GOAL)
        p._reset()
        assert p.total == 0.0 and p.last_reward == 0.0 and p.outcome is None


class TestOutcome:
    def test_lava_death_is_labelled(self):
        p = make_player()
        walk_onto(p, p.core.lava_positions[0])
        assert p.outcome == "lava"

    def test_time_limit_is_labelled(self):
        p = make_player(max_steps=3)
        for _ in range(3):
            p._step(0)
        assert p.outcome == "time"

    def test_alive_while_playing(self):
        p = make_player()
        p._step(0)
        assert p.outcome is None

    def test_reaching_the_goal_does_not_end_the_episode(self):
        p = make_player()
        walk_onto(p, GOAL)
        assert p.outcome is None


class TestSessionCounters:
    def test_counts_a_successful_episode(self):
        p = make_player()
        walk_onto(p, GOAL)
        walk_onto(p, p.core.lava_positions[0])
        assert (p.episodes, p.successes) == (1, 1)

    def test_death_without_the_goal_is_not_a_success(self):
        p = make_player()
        walk_onto(p, p.core.lava_positions[0])
        assert (p.episodes, p.successes) == (1, 0)


class TestKeys:
    def test_escape_stops_the_loop(self):
        assert make_player()._on_key(pygame.K_ESCAPE) is False

    def test_arrow_steps(self):
        p = make_player()
        p._on_key(pygame.K_UP)
        assert p.core.step_count == 1

    def test_grid_freezes_after_the_episode_ends(self):
        p = make_player()
        walk_onto(p, p.core.lava_positions[0])
        frozen = p.core.step_count
        for key in (pygame.K_UP, pygame.K_LEFT, pygame.K_SPACE):
            assert p._on_key(key) is True
        assert p.core.step_count == frozen

    def test_r_starts_a_new_episode(self):
        p = make_player()
        walk_onto(p, p.core.lava_positions[0])
        p._on_key(pygame.K_r)
        assert p.outcome is None and p.core.step_count == 0


class TestWorksAcrossTheFamily:
    @pytest.mark.parametrize("cls", [GoalGrid, LavaGrid])
    def test_goal_probe_works_without_goal_reached(self, cls):
        """GoalGrid has no `goal_reached`; the probe falls back to the current cell."""
        p = make_player(cls)
        assert p._goal_reached() is False
        walk_onto(p, GOAL)
        assert p._goal_reached() is True

    @pytest.mark.parametrize("cls", [GoalGrid, LavaGrid])
    def test_steps_without_error(self, cls):
        p = make_player(cls)
        p._step(FORWARD)
