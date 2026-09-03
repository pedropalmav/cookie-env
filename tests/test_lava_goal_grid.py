import gymnasium as gym
import pytest
from minigrid.wrappers import RGBImgObsWrapper

import cookie_env  # noqa: F401  — registers the env ids
from cookie_env.envs.lava_grid import LavaGrid
from cookie_env.envs.lava_goal_grid import LavaGoalGrid, make_lava_goal_grid_env

FORWARD = 2
LEFT = 0

GOAL = (7, 7)


def walk_onto(env, cell):
    """Teleport the agent next to `cell`, face it, and step forward onto it."""
    x, y = cell
    env.agent_pos = (x - 1, y)
    env.agent_dir = 0  # east
    return env.step(FORWARD)


def make_env(**kwargs):
    kwargs.setdefault("size", 9)
    kwargs.setdefault("n_lava", 1)
    kwargs.setdefault("goal_pos", GOAL)
    kwargs.setdefault("agent_start_pos", (1, 1))
    return LavaGoalGrid(**kwargs)


class TestConstruction:
    def test_is_a_lava_grid(self):
        assert issubclass(LavaGoalGrid, LavaGrid)

    def test_default_goal_reward(self):
        assert make_env().goal_reward == 1.0

    def test_lava_still_placed(self):
        env = make_env(n_lava=3)
        env.reset(seed=0)
        assert len(env.lava_positions) == 3


class TestRewardScheme:
    def test_ordinary_step_is_zero(self):
        """The -1 per step of GoalGrid is gone — this is the whole point."""
        env = make_env()
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(LEFT)
        assert reward == 0.0

    def test_reaching_the_goal_pays(self):
        env = make_env()
        env.reset(seed=0)
        _, reward, _, _, info = walk_onto(env, GOAL)
        assert reward == 1.0
        assert info["goal"] is True

    def test_goal_reward_is_configurable(self):
        env = make_env(goal_reward=7.5)
        env.reset(seed=0)
        _, reward, _, _, _ = walk_onto(env, GOAL)
        assert reward == 7.5

    def test_dying_pays_nothing_by_default(self):
        env = make_env()
        env.reset(seed=0)
        _, reward, terminated, _, _ = walk_onto(env, env.lava_positions[0])
        assert terminated is True
        assert reward == 0.0

    def test_no_reward_is_ever_negative(self):
        """Death must never beat survival, which is why this env exists."""
        env = make_env(size=7, agent_start_pos=None, goal_pos=None, max_steps=50)
        for seed in range(20):
            env.reset(seed=seed)
            for action in (FORWARD, LEFT, FORWARD, FORWARD, LEFT, FORWARD):
                _, reward, terminated, _, _ = env.step(action)
                assert reward >= 0.0
                if terminated:
                    break

    def test_inherited_lava_penalty_still_applies(self):
        """Kept as an escape hatch, even though it breaks the non-negative scheme."""
        env = make_env(lava_penalty=10.0)
        env.reset(seed=0)
        _, reward, terminated, _, _ = walk_onto(env, env.lava_positions[0])
        assert terminated is True
        assert reward == -10.0


class TestGoalRewardPaidOnce:
    def test_staying_on_the_goal_pays_nothing_extra(self):
        env = make_env()
        env.reset(seed=0)
        walk_onto(env, GOAL)
        for _ in range(5):
            _, reward, _, _, info = env.step(LEFT)  # turn in place, stays on goal
            assert info["goal"] is True
            assert reward == 0.0

    def test_leaving_and_returning_pays_nothing_extra(self):
        env = make_env()
        env.reset(seed=0)
        assert walk_onto(env, GOAL)[1] == 1.0
        env.agent_pos = (1, 1)
        assert walk_onto(env, GOAL)[1] == 0.0

    def test_episode_return_is_a_success_flag(self):
        env = make_env(max_steps=100)
        env.reset(seed=0)
        total = walk_onto(env, GOAL)[1]
        for _ in range(20):
            total += env.step(LEFT)[1]
        assert total == 1.0

    def test_goal_reached_tracks_the_episode(self):
        env = make_env()
        env.reset(seed=0)
        assert env.goal_reached is False
        walk_onto(env, GOAL)
        assert env.goal_reached is True

    def test_reset_clears_the_payment(self):
        env = make_env()
        env.reset(seed=0)
        walk_onto(env, GOAL)
        env.reset(seed=1)
        assert env.goal_reached is False
        assert walk_onto(env, GOAL)[1] == 1.0


class TestTermination:
    def test_lava_still_terminates(self):
        env = make_env()
        env.reset(seed=0)
        _, _, terminated, _, info = walk_onto(env, env.lava_positions[0])
        assert terminated is True
        assert info["lava"] is True

    def test_reaching_the_goal_does_not_terminate(self):
        """Inherited from GoalGrid on purpose — only death is terminal."""
        env = make_env()
        env.reset(seed=0)
        _, _, terminated, truncated, _ = walk_onto(env, GOAL)
        assert terminated is False
        assert truncated is False

    def test_truncates_at_max_steps(self):
        env = make_env(max_steps=3)
        env.reset(seed=0)
        truncated = False
        for _ in range(3):
            _, _, _, truncated, _ = env.step(LEFT)
        assert truncated is True


class TestInheritedBehaviour:
    def test_mission_still_present(self):
        obs, _ = make_env().reset(seed=0)
        assert "mission" in obs

    def test_lava_never_lands_on_the_goal(self):
        env = make_env(size=7, n_lava=3, goal_pos=(3, 3))
        for seed in range(30):
            env.reset(seed=seed)
            assert (3, 3) not in env.lava_positions

    def test_random_goal_moves_across_resets(self):
        env = make_env(size=9, goal_pos=None)
        seen = set()
        for seed in range(20):
            env.reset(seed=seed)
            seen.add(tuple(env.goal_position))
        assert len(seen) > 1


class TestFactoryAndRegistration:
    def test_factory_wraps_in_rgb_obs_wrapper(self):
        assert isinstance(make_lava_goal_grid_env(size=7), RGBImgObsWrapper)

    def test_factory_passes_kwargs(self):
        env = make_lava_goal_grid_env(size=9, n_lava=2, goal_reward=3.0)
        env.reset(seed=0)
        assert env.unwrapped.n_lava == 2
        assert env.unwrapped.goal_reward == 3.0

    def test_gym_make(self):
        env = gym.make("LavaGoalGrid-v0")
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(LEFT)
        assert reward == 0.0
        assert env.unwrapped.n_lava == 1
        env.close()

    def test_registered_id_leaves_goal_random(self):
        env = gym.make("LavaGoalGrid-v0")
        env.reset(seed=0)
        assert env.unwrapped.goal_pos is None
        env.close()
