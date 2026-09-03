from minigrid.wrappers import RGBImgObsWrapper

from cookie_env.envs.lava_grid import LavaGrid


class LavaGoalGrid(LavaGrid):
    """`LavaGrid` with a never-negative reward: +``goal_reward`` for reaching the
    green square, 0 everywhere else.

    This is the answer to the incentive `LavaGrid` documents and leaves open.
    With -1 per step, ending the episode is worth more than surviving it, so lava
    is attractive to any agent that can perceive termination. Here the per-step
    reward is 0 and dying is worth 0, so death is never better than staying
    alive — it just forfeits whatever the agent had not collected yet.

    Reward, in full:

    - every step off the square: 0
    - first step onto the square: ``goal_reward`` (1.0 by default)
    - stepping into lava: 0, and ``terminated=True``

    **The goal reward is paid once per episode.** Reaching the square does not
    end the episode (that convention is inherited from `GoalGrid` on purpose),
    so an agent parked on the square would collect the bonus on every step and
    the return would degenerate into a dwell-time count. Paying once keeps the
    episode return in ``{0, goal_reward}``: it is a success flag, not the
    step-count proxy that `GoalGrid` returns.

    That flag is the trade-off to be aware of before training on this: the
    reward is sparse and carries no gradient toward the square, where
    `GoalGrid`'s -1 per step at least rewards getting there sooner. Exploration
    has to come from somewhere else — an intrinsic bonus, a curriculum, or the
    consumer's own goal-conditioned signal.

    ``lava_penalty`` is inherited and still subtracted on the terminating step.
    It defaults to 0, which is what keeps the scheme non-negative; setting it
    re-introduces exactly the sign problem this class exists to avoid, so it is
    left as a deliberate choice rather than removed.
    """

    def __init__(self, *args, goal_reward: float = 1.0, **kwargs):
        self.goal_reward = goal_reward
        self._goal_paid = False
        super().__init__(*args, **kwargs)

    def _on_goal(self):
        cell = self.grid.get(*self.agent_pos)
        return cell is not None and cell.type == "goal"

    def _reward(self):
        # Overrides GoalGrid's -1/0: this is the hook GoalGrid.step calls, so
        # replacing it keeps LavaGrid's lava_penalty subtraction working.
        if self._goal_paid or not self._on_goal():
            return 0.0
        return self.goal_reward

    @property
    def goal_reached(self):
        """Whether the square has been reached at any point this episode."""
        return self._goal_paid

    def reset(self, **kwargs):
        self._goal_paid = False
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        on_goal = self._on_goal()
        info["goal"] = on_goal
        # Marked after super().step(), so the _reward() call inside it saw the
        # pre-step flag: the bonus lands on the step that arrives at the square
        # and never again.
        self._goal_paid = self._goal_paid or on_goal
        return obs, reward, terminated, truncated, info


def make_lava_goal_grid_env(
    size: int = 10,
    agent_start_pos: tuple[int, int] | None = (1, 1),
    agent_start_dir: int = 0,
    max_steps: int = 100,
    goal_pos: tuple[int, int] | None = None,
    n_lava: int = 1,
    lava_penalty: float = 0.0,
    goal_reward: float = 1.0,
    **kwargs,
):
    env = LavaGoalGrid(
        size=size,
        agent_start_pos=agent_start_pos,
        agent_start_dir=agent_start_dir,
        goal_pos=goal_pos,
        max_steps=max_steps,
        n_lava=n_lava,
        lava_penalty=lava_penalty,
        goal_reward=goal_reward,
        **kwargs,
    )
    return RGBImgObsWrapper(env)


if __name__ == "__main__":
    # python -m cookie_env.envs.lava_goal_grid
    from cookie_env.utils.play import play

    size = 10
    # rgb_array, not human: the viewer owns the window so it can draw the HUD.
    env = LavaGoalGrid(size=size, max_steps=4 * size, n_lava=3, render_mode="rgb_array")

    play(env)
