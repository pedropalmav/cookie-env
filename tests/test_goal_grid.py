import re

import gymnasium as gym
import numpy as np
import pytest
from minigrid.wrappers import RGBImgObsWrapper

import cookie_env  # noqa: F401  — registers the env ids
from cookie_env.envs.goal_grid import DIRECTIONS, GoalGrid, make_goal_grid_env

FIXED = (3, 1)


@pytest.fixture(params=["fixed", "random"])
def variant(request):
    """The two goal modes, so shared behaviour is asserted for both."""
    return request.param


def make_env(variant, size=7, **kwargs):
    goal_pos = FIXED if variant == "fixed" else None
    return GoalGrid(size=size, goal_pos=goal_pos, **kwargs)


class TestInit:
    def test_size_stored(self, variant):
        assert make_env(variant, size=5).size == 5

    def test_agent_start_pos_stored(self, variant):
        assert make_env(variant, agent_start_pos=(1, 1)).agent_start_pos == (1, 1)

    def test_gen_mission(self):
        assert GoalGrid._gen_mission() == "reach goal"

    def test_defaults_to_random_goal(self):
        assert GoalGrid(size=5).goal_pos is None


class TestGoalPlacement:
    def test_fixed_goal_is_stable_across_resets(self):
        env = GoalGrid(size=7, goal_pos=FIXED)
        seen = set()
        for seed in range(10):
            env.reset(seed=seed)
            seen.add(tuple(env.goal_position))
        assert seen == {FIXED}

    def test_random_goal_moves_across_resets(self):
        env = GoalGrid(size=7, goal_pos=None)
        seen = set()
        for seed in range(20):
            env.reset(seed=seed)
            seen.add(tuple(env.goal_position))
        assert len(seen) > 1

    def test_goal_object_is_on_the_grid(self, variant):
        env = make_env(variant)
        env.reset(seed=0)
        cell = env.grid.get(*env.goal_position)
        assert cell is not None and cell.type == "goal"

    def test_goal_position_is_concrete_even_when_configured_none(self):
        """`goal_pos` may stay None; `goal_position` must always be a real cell."""
        env = GoalGrid(size=7, goal_pos=None)
        env.reset(seed=0)
        assert env.goal_pos is None
        x, y = env.goal_position
        assert 1 <= x <= 5 and 1 <= y <= 5

    def test_assigning_goal_pos_pins_the_square(self):
        """The contract consumers rely on to render a chosen layout: set
        `goal_pos`, reset, and the square lands exactly there."""
        env = GoalGrid(size=7, goal_pos=None)
        env.reset(seed=0)
        env.goal_pos = (5, 5)
        env.reset(seed=1)
        assert tuple(env.goal_position) == (5, 5)
        cell = env.grid.get(5, 5)
        assert cell is not None and cell.type == "goal"


class TestAgentPlacement:
    def test_explicit_start_pos(self, variant):
        env = make_env(variant, agent_start_pos=(1, 1), agent_start_dir=2)
        env.reset(seed=0)
        assert env.agent_pos == (1, 1)
        assert env.agent_dir == 2

    def test_none_start_pos_places_randomly(self, variant):
        env = make_env(variant, size=5, agent_start_pos=None)
        env.reset(seed=0)
        x, y = env.agent_pos
        assert 0 <= x < 5 and 0 <= y < 5


class TestMissions:
    def test_build_mission_format(self, variant):
        env = make_env(variant, size=5, agent_start_pos=(1, 1), agent_start_dir=0)
        env.reset(seed=0)
        ax, ay = env.agent_pos
        gx, gy = env.goal_position
        expected = f"agent at ({ax},{ay}) facing {DIRECTIONS[env.agent_dir]}. goal at ({gx},{gy})"
        assert env._build_mission() == expected

    def test_random_mission_has_all_fields(self, variant):
        env = make_env(variant)
        env.reset(seed=0)
        mission = env.random_mission(rng=np.random.RandomState(42))
        assert "agent at" in mission and "goal at" in mission and "facing" in mission

    def test_random_mission_without_rng(self, variant):
        env = make_env(variant)
        env.reset(seed=0)
        assert "agent at" in env.random_mission()

    def test_random_mission_positions_within_interior(self, variant):
        env = make_env(variant, size=7)
        env.reset(seed=0)
        rng = np.random.RandomState(0)
        for _ in range(20):
            for pos in (int(n) for n in re.findall(r"\d+", env.random_mission(rng=rng))):
                assert 1 <= pos <= 5

    def test_random_mission_keeps_a_fixed_goal_fixed(self):
        env = GoalGrid(size=7, goal_pos=FIXED)
        env.reset(seed=0)
        rng = np.random.RandomState(0)
        for _ in range(20):
            assert f"goal at ({FIXED[0]},{FIXED[1]})" in env.random_mission(rng=rng)

    def test_random_mission_moves_the_goal_when_random(self):
        env = GoalGrid(size=7, goal_pos=None)
        env.reset(seed=0)
        rng = np.random.RandomState(0)
        goals = {m.group(0) for m in (re.search(r"goal at \(\d+,\d+\)", env.random_mission(rng=rng)) for _ in range(20))}
        assert len(goals) > 1


class TestResetAndStep:
    def test_reset_adds_mission(self, variant):
        obs, _ = make_env(variant).reset(seed=0)
        assert "mission" in obs

    def test_step_adds_mission(self, variant):
        env = make_env(variant)
        env.reset(seed=0)
        obs, _, _, _, _ = env.step(0)
        assert "mission" in obs

    def test_terminated_always_false(self, variant):
        """Success does not end the episode — it only stops the -1 per step."""
        env = make_env(variant)
        env.reset(seed=0)
        _, _, terminated, _, _ = env.step(0)
        assert terminated is False

    def test_reward_off_goal(self, variant):
        env = make_env(variant, size=5, agent_start_pos=(1, 1))
        env.reset(seed=0)
        if tuple(env.agent_pos) != tuple(env.goal_position):
            assert env._reward() == -1

    def test_reward_on_goal(self, variant):
        env = make_env(variant, size=5)
        env.reset(seed=0)
        env.agent_pos = env.goal_position
        assert env._reward() == 0

    def test_truncates_at_max_steps(self, variant):
        env = make_env(variant, max_steps=3)
        env.reset(seed=0)
        truncated = False
        for _ in range(3):
            _, _, _, truncated, _ = env.step(0)
        assert truncated is True


class TestFactory:
    def test_wraps_in_rgb_obs_wrapper(self, variant):
        goal_pos = FIXED if variant == "fixed" else None
        assert isinstance(make_goal_grid_env(size=5, goal_pos=goal_pos), RGBImgObsWrapper)

    def test_kwargs_passed_through(self, variant):
        goal_pos = FIXED if variant == "fixed" else None
        env = make_goal_grid_env(size=5, goal_pos=goal_pos, agent_start_dir=3)
        env.reset(seed=0)
        assert env.unwrapped.agent_start_dir == 3

    def test_observation_has_rgb_image(self, variant):
        goal_pos = FIXED if variant == "fixed" else None
        obs, _ = make_goal_grid_env(size=5, goal_pos=goal_pos).reset(seed=0)
        assert obs["image"].ndim == 3 and obs["image"].shape[-1] == 3


class TestRegisteredIds:
    @pytest.mark.parametrize("env_id", ["GoalGrid-v0", "GoalGrid-random-v0"])
    def test_gym_make(self, env_id):
        env = gym.make(env_id)
        env.reset(seed=0)
        env.step(0)
        env.close()

    def test_fixed_id_pins_the_square(self):
        env = gym.make("GoalGrid-v0")
        env.reset(seed=0)
        assert tuple(env.unwrapped.goal_position) == (8, 1)

    def test_random_id_leaves_goal_unconfigured(self):
        env = gym.make("GoalGrid-random-v0")
        env.reset(seed=0)
        assert env.unwrapped.goal_pos is None
