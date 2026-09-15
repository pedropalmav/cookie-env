import gymnasium as gym
import pytest
from minigrid.core.world_object import Lava
from minigrid.wrappers import RGBImgObsWrapper

import cookie_env  # noqa: F401  — registers the env ids
from cookie_env.envs.goal_grid import GoalGrid
from cookie_env.envs.lava_grid import LavaGrid, make_lava_grid_env

FORWARD = 2


def walk_onto(env, cell):
    """Teleport the agent next to `cell`, face it, and step forward onto it."""
    x, y = cell
    env.agent_pos = (x - 1, y)
    env.agent_dir = 0  # east
    return env.step(FORWARD)


class TestConstruction:
    def test_is_a_goal_grid(self):
        assert issubclass(LavaGrid, GoalGrid)

    def test_default_has_one_lava_cell(self):
        env = LavaGrid(size=7)
        env.reset(seed=0)
        assert env.n_lava == 1
        assert len(env.lava_positions) == 1

    @pytest.mark.parametrize("n_lava", [0, 1, 3])
    def test_n_lava_is_honoured(self, n_lava):
        env = LavaGrid(size=9, n_lava=n_lava)
        env.reset(seed=0)
        assert len(env.lava_positions) == n_lava

    def test_lava_objects_are_on_the_grid(self):
        env = LavaGrid(size=9, n_lava=3)
        env.reset(seed=0)
        for pos in env.lava_positions:
            assert isinstance(env.grid.get(*pos), Lava)

    def test_lava_moves_across_resets(self):
        env = LavaGrid(size=9)
        seen = {tuple(env.reset(seed=s)[0] and env.lava_positions[0]) for s in range(20)}
        assert len(seen) > 1


class TestPlacementSafety:
    def test_lava_never_lands_on_the_goal(self):
        env = LavaGrid(size=7, n_lava=3, goal_pos=(3, 3))
        for seed in range(30):
            env.reset(seed=seed)
            assert (3, 3) not in env.lava_positions

    def test_lava_never_lands_on_the_agent(self):
        env = LavaGrid(size=7, n_lava=3, agent_start_pos=(1, 1))
        for seed in range(30):
            env.reset(seed=seed)
            assert tuple(env.agent_pos) not in env.lava_positions

    def test_agent_does_not_start_dead(self):
        env = LavaGrid(size=7, n_lava=3, agent_start_pos=None)
        for seed in range(30):
            env.reset(seed=seed)
            assert not env._on_lava()


class TestPinnedLava:
    """`lava_pos` is to lava what `goal_pos` is to the green square."""

    def test_pinned_cells_are_used_verbatim(self):
        env = LavaGrid(size=9, goal_pos=(7, 7), agent_start_pos=(1, 1), lava_pos=[(3, 3), (4, 5)])
        env.reset(seed=0)
        assert env.lava_positions == [(3, 3), (4, 5)]

    def test_pinned_cells_survive_resets(self):
        env = LavaGrid(size=9, goal_pos=(7, 7), agent_start_pos=(1, 1), lava_pos=[(3, 3)])
        for seed in range(10):
            env.reset(seed=seed)
            assert env.lava_positions == [(3, 3)]

    def test_n_lava_is_derived_from_the_pinned_list(self):
        """Rather than trusted: a caller passing both can't make them disagree."""
        env = LavaGrid(size=9, n_lava=7, goal_pos=(7, 7), agent_start_pos=(1, 1), lava_pos=[(3, 3), (4, 5)])
        env.reset(seed=0)
        assert env.n_lava == 2

    def test_n_lava_follows_a_later_assignment(self):
        env = LavaGrid(size=9, n_lava=1, goal_pos=(7, 7), agent_start_pos=(1, 1))
        env.lava_pos = [(3, 3), (4, 5), (5, 3)]
        env.reset(seed=0)
        assert env.n_lava == 3

    def test_pinned_lava_is_on_the_grid(self):
        env = LavaGrid(size=9, goal_pos=(7, 7), agent_start_pos=(1, 1), lava_pos=[(3, 3), (4, 5)])
        env.reset(seed=0)
        for pos in env.lava_positions:
            assert isinstance(env.grid.get(*pos), Lava)

    def test_pinned_lava_still_terminates(self):
        env = LavaGrid(size=9, goal_pos=(7, 7), agent_start_pos=(1, 1), lava_pos=[(3, 3)])
        env.reset(seed=0)
        _, _, terminated, _, info = walk_onto(env, (3, 3))
        assert terminated is True
        assert info["lava"] is True

    def test_assigning_lava_pos_repins_on_reset(self):
        """The documented way to pin a layout: assign, then reset."""
        env = LavaGrid(size=9, goal_pos=(7, 7), agent_start_pos=(1, 1))
        env.reset(seed=0)
        env.lava_pos = [(5, 5)]
        env.reset(seed=0)
        assert env.lava_positions == [(5, 5)]

    def test_random_agent_never_spawns_on_pinned_lava(self):
        """Pinned lava goes down before the agent, so `place_agent` avoids it."""
        env = LavaGrid(size=7, agent_start_pos=None, goal_pos=(5, 5), lava_pos=[(2, 2), (3, 3), (2, 3)])
        for seed in range(40):
            env.reset(seed=seed)
            assert tuple(env.agent_pos) not in env.lava_positions
            assert not env._on_lava()

    def test_empty_list_means_no_lava(self):
        env = LavaGrid(size=9, goal_pos=(7, 7), agent_start_pos=(1, 1), lava_pos=[])
        env.reset(seed=0)
        assert env.lava_positions == []
        assert env.n_lava == 0


class TestPinnedLavaConflicts:
    """A pinned cell the layout cannot honour is a caller error, not a silent fix."""

    def test_cell_on_the_goal_raises(self):
        env = LavaGrid(size=9, goal_pos=(7, 7), agent_start_pos=(1, 1), lava_pos=[(7, 7)])
        with pytest.raises(ValueError, match="collides with the goal"):
            env.reset(seed=0)

    def test_cell_on_the_agent_start_raises(self):
        env = LavaGrid(size=9, goal_pos=(7, 7), agent_start_pos=(1, 1), lava_pos=[(1, 1)])
        with pytest.raises(ValueError, match="collides with agent_start_pos"):
            env.reset(seed=0)

    @pytest.mark.parametrize("cell", [(0, 3), (3, 0), (8, 3), (3, 8)])
    def test_cell_on_a_wall_raises(self, cell):
        env = LavaGrid(size=9, goal_pos=(7, 7), agent_start_pos=(1, 1), lava_pos=[cell])
        with pytest.raises(ValueError, match="outside the grid interior"):
            env.reset(seed=0)


class TestTermination:
    def test_stepping_into_lava_terminates(self):
        env = LavaGrid(size=9, n_lava=1, goal_pos=(7, 7))
        env.reset(seed=0)
        _, _, terminated, _, info = walk_onto(env, env.lava_positions[0])
        assert terminated is True
        assert info["lava"] is True

    def test_ordinary_step_does_not_terminate(self):
        """Only lava is terminal — reaching the goal is not."""
        env = LavaGrid(size=9, n_lava=1, goal_pos=(7, 7), agent_start_pos=(1, 1))
        env.reset(seed=0)
        _, _, terminated, _, _ = env.step(FORWARD)
        assert terminated is False

    def test_reaching_the_goal_does_not_terminate(self):
        env = LavaGrid(size=9, n_lava=1, goal_pos=(7, 7))
        env.reset(seed=0)
        _, reward, terminated, _, _ = walk_onto(env, (7, 7))
        assert terminated is False
        assert reward == 0

    def test_default_penalty_leaves_the_step_reward_alone(self):
        env = LavaGrid(size=9, n_lava=1, goal_pos=(7, 7))
        env.reset(seed=0)
        _, reward, _, _, _ = walk_onto(env, env.lava_positions[0])
        assert reward == -1

    def test_lava_penalty_is_subtracted_on_death(self):
        env = LavaGrid(size=9, n_lava=1, goal_pos=(7, 7), lava_penalty=100.0)
        env.reset(seed=0)
        _, reward, terminated, _, _ = walk_onto(env, env.lava_positions[0])
        assert terminated is True
        assert reward == -101

    def test_penalty_not_applied_when_alive(self):
        env = LavaGrid(size=9, n_lava=1, goal_pos=(7, 7), agent_start_pos=(1, 1), lava_penalty=100.0)
        env.reset(seed=0)
        _, reward, terminated, _, _ = env.step(FORWARD)
        assert terminated is False
        assert reward == -1


class TestInheritedBehaviour:
    def test_mission_still_present(self):
        obs, _ = LavaGrid(size=7).reset(seed=0)
        assert "mission" in obs

    def test_fixed_goal_still_pinned(self):
        env = LavaGrid(size=7, goal_pos=(3, 1))
        seen = set()
        for seed in range(10):
            env.reset(seed=seed)
            seen.add(tuple(env.goal_position))
        assert seen == {(3, 1)}

    def test_truncates_at_max_steps(self):
        env = LavaGrid(size=9, max_steps=3, goal_pos=(7, 7), agent_start_pos=(1, 1))
        env.reset(seed=0)
        truncated = False
        for _ in range(3):
            _, _, _, truncated, _ = env.step(0)  # turn left, never reaches lava
        assert truncated is True


class TestFactoryAndRegistration:
    def test_factory_wraps_in_rgb_obs_wrapper(self):
        assert isinstance(make_lava_grid_env(size=7), RGBImgObsWrapper)

    def test_factory_passes_lava_kwargs(self):
        env = make_lava_grid_env(size=9, n_lava=2, lava_penalty=5.0)
        env.reset(seed=0)
        assert env.unwrapped.n_lava == 2
        assert env.unwrapped.lava_penalty == 5.0

    def test_factory_passes_lava_pos(self):
        env = make_lava_grid_env(size=9, goal_pos=(7, 7), agent_start_pos=(1, 1), lava_pos=[(3, 3)])
        env.reset(seed=0)
        assert env.unwrapped.lava_positions == [(3, 3)]

    def test_gym_make(self):
        env = gym.make("LavaGrid-v0")
        env.reset(seed=0)
        env.step(0)
        assert env.unwrapped.n_lava == 1
        env.close()
