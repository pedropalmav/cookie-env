import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import RGBImgObsWrapper

DIRECTIONS = {0: "east", 1: "south", 2: "west", 3: "north"}


class GoalGrid(MiniGridEnv):
    """Empty room where the agent must reach a green square.

    `goal_pos` selects the two variants that used to be separate classes:

    - a ``(x, y)`` tuple keeps the square at that cell for every episode
    - ``None`` places it at a uniformly random free cell on each reset

    This mirrors the convention MiniGrid already uses for the agent, where
    ``agent_start_pos=None`` means "spawn anywhere", so one knob reads the same
    way for both entities.

    Two attributes track the goal on purpose. ``goal_pos`` is the *configured*
    value and may be ``None``; ``_goal_pos`` is the cell actually used by the
    current episode and is always concrete. Consumers that pin the layout (for
    example a generator that renders the observation of a synthetic state) assign
    to ``goal_pos`` and call ``reset()``; consumers that read where the square
    ended up use the ``goal_position`` property.

    The episode never terminates on success: reaching the square yields reward 0
    instead of -1 and the episode runs to the time limit, so return is a
    step-count proxy rather than a success flag.
    """

    def __init__(
        self,
        size: int = 10,
        agent_start_pos: tuple[int, int] | None = (1, 1),
        agent_start_dir: int = 0,
        max_steps: int = 100,
        render_mode: str = "rgb_array",
        goal_pos: tuple[int, int] | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = goal_pos
        self._goal_pos = goal_pos
        self.size = size

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            render_mode=render_mode,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "reach goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self._put_goal()
        self._put_agent()

    def _put_goal(self):
        if self.goal_pos is None:
            self._goal_pos = self.place_obj(Goal())
        else:
            self._goal_pos = tuple(self.goal_pos)
            self.put_obj(Goal(), *self._goal_pos)

    def _put_agent(self):
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def _build_mission(self):
        ax, ay = self.agent_pos
        direction = DIRECTIONS[self.agent_dir]
        gx, gy = self._goal_pos
        return f"agent at ({ax},{ay}) facing {direction}. goal at ({gx},{gy})"

    def random_mission(self, rng: np.random.RandomState = None) -> str:
        """Return a mission string with randomly sampled [ax, ay, direction, gx, gy].

        All positions are sampled uniformly from the grid interior (1 to size-2),
        independently — agent and goal may coincide, matching the training
        distribution. When the goal is configured to a fixed cell it stays fixed
        here too, so the sampled distribution matches what the env produces.
        """
        if rng is None:
            rng = np.random.RandomState()
        interior = np.arange(1, self.size - 1)
        ax, ay = rng.choice(interior), rng.choice(interior)
        if self.goal_pos is None:
            gx, gy = rng.choice(interior), rng.choice(interior)
        else:
            gx, gy = self.goal_pos
        direction = DIRECTIONS[rng.randint(0, 4)]
        return f"agent at ({ax},{ay}) facing {direction}. goal at ({gx},{gy})"

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs["mission"] = self._build_mission()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        reward = self._reward()
        obs["mission"] = self._build_mission()
        # We use truncated to signal episode end instead of terminated
        terminated = False
        return obs, reward, terminated, truncated, info

    def _reward(self):
        agent_cell = self.grid.get(*self.agent_pos)
        return 0 if agent_cell is not None and agent_cell.type == "goal" else -1

    @property
    def goal_position(self):
        """Current episode's goal cell, as a uniform interface for consumers that
        need to know where the square actually is (fixed or randomly placed)."""
        return self._goal_pos


def make_goal_grid_env(
    size: int = 10,
    agent_start_pos: tuple[int, int] | None = (1, 1),
    agent_start_dir: int = 0,
    max_steps: int = 100,
    goal_pos: tuple[int, int] | None = None,
    **kwargs,
):
    env = GoalGrid(
        size=size,
        agent_start_pos=agent_start_pos,
        agent_start_dir=agent_start_dir,
        goal_pos=goal_pos,
        max_steps=max_steps,
        **kwargs,
    )
    return RGBImgObsWrapper(env)


if __name__ == "__main__":
    # python -m cookie_env.envs.goal_grid
    from minigrid.manual_control import ManualControl

    size = 10
    env = make_goal_grid_env(size=size, max_steps=2 * size, render_mode="human")

    manual_control = ManualControl(env)
    manual_control.start()
