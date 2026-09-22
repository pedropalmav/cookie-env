from minigrid.core.world_object import Lava
from minigrid.wrappers import RGBImgObsWrapper

from cookie_env.envs.goal_grid import GoalGrid


class LavaGrid(GoalGrid):
    """`GoalGrid` plus lava cells that end the episode on contact, with a
    never-negative reward: +``goal_reward`` once for reaching the green square,
    0 everywhere else.

    This is the episodic counterpart of `GoalGrid`: reaching the green square
    still does *not* terminate, but stepping into lava sets ``terminated=True``.
    Death is the only terminal, which matches environments like Crafter where
    there is no single goal cell but dying does end the episode.

    Reward, in full:

    - every step off the square: 0
    - first step onto the square: ``goal_reward`` (1.0 by default)
    - stepping into lava: 0 (minus ``lava_penalty``), and ``terminated=True``

    **Why not `GoalGrid`'s -1 per step.** With a negative per-step reward,
    terminating is worth more than surviving, because ending the episode stops
    the accumulation of -1s. Measured with a random policy under a 0.997
    discount, dying scored -136 against -309 for running to the time limit, so
    any agent that can perceive termination has an incentive to seek the lava.
    With 0 per step and 0 for dying, death is never better than staying alive —
    it just forfeits whatever the agent had not collected yet (0.31 dying vs
    0.47 surviving, same measurement).

    **The goal reward is paid once per episode.** Reaching the square does not
    end the episode, so an agent parked on the square would otherwise collect
    the bonus on every step and the return would degenerate into a dwell-time
    count. Paying once keeps the episode return in ``{0, goal_reward}``: it is a
    success flag, not the step-count proxy that `GoalGrid` returns.

    That flag is the trade-off to be aware of before training on this: the
    reward is sparse and carries no gradient toward the square, where
    `GoalGrid`'s -1 per step at least rewards getting there sooner. Exploration
    has to come from somewhere else — an intrinsic bonus, a curriculum, or the
    consumer's own goal-conditioned signal. Some consumers do not use this reward
    at all: her-dream overwrites ``reward`` with a goal-conditioned latent reward
    before the transition reaches its buffer, so for it only the lava terminal
    matters.

    ``lava_penalty`` is subtracted on the terminating step. It defaults to 0,
    which is what keeps the scheme non-negative; setting it re-introduces exactly
    the sign problem above, so it is left as a deliberate choice rather than
    removed.

    ``lava_pos`` selects the same two variants ``goal_pos`` does, one level up:

    - ``None`` resamples ``n_lava`` cells on every reset (the default)
    - a list of ``(x, y)`` keeps the lava at those cells for every episode

    As with ``goal_pos``, the configured value may be ``None`` while
    ``lava_positions`` is always the concrete list for the current episode, and a
    consumer that needs to pin a layout assigns to ``lava_pos`` and calls
    ``reset()``. ``n_lava`` then only describes how many cells to resample; when
    ``lava_pos`` is given it is derived from that list, so the two can never
    disagree.

    Resampled lava is placed after the goal and the agent, so it can never spawn on
    either. Pinned lava goes down *before* the agent instead — otherwise an agent
    spawned at random (``agent_start_pos=None``) could land on a cell the caller
    already claimed for lava. A pinned cell that collides with the goal, with a
    configured agent start, or with a wall is a caller error and raises.
    """

    def __init__(
        self,
        *args,
        n_lava: int = 1,
        lava_penalty: float = 0.0,
        goal_reward: float = 1.0,
        lava_pos: list[tuple[int, int]] | None = None,
        **kwargs,
    ):
        self.lava_pos = None if lava_pos is None else [(int(x), int(y)) for x, y in lava_pos]
        self._n_lava = n_lava
        self.lava_penalty = lava_penalty
        self.goal_reward = goal_reward
        self._goal_paid = False
        self._lava_positions: list[tuple[int, int]] = []
        super().__init__(*args, **kwargs)

    def _gen_grid(self, width, height):
        # Goal, then (pinned lava), then agent — see _put_agent.
        super()._gen_grid(width, height)
        if self.lava_pos is None:
            # place_obj skips occupied cells and the agent's cell, so resampled lava
            # cannot land on either.
            self._lava_positions = [tuple(self.place_obj(Lava())) for _ in range(self.n_lava)]

    def _put_agent(self):
        # Pinned lava is placed here, ahead of the agent: `place_agent` skips
        # occupied cells, so putting the lava down first is what keeps a randomly
        # spawned agent off it. Resampled lava cannot use this hook — it has to see
        # the agent's cell in order to avoid it — hence the split with _gen_grid.
        if self.lava_pos is not None:
            self._put_pinned_lava()
        super()._put_agent()

    def _put_pinned_lava(self):
        self._lava_positions = []
        for x, y in self.lava_pos:
            if not (1 <= x <= self.grid.width - 2 and 1 <= y <= self.grid.height - 2):
                raise ValueError(f"lava_pos cell {(x, y)} is outside the grid interior")
            if (x, y) == tuple(self._goal_pos):
                raise ValueError(f"lava_pos cell {(x, y)} collides with the goal")
            if self.agent_start_pos is not None and (x, y) == tuple(self.agent_start_pos):
                raise ValueError(f"lava_pos cell {(x, y)} collides with agent_start_pos")
            self.put_obj(Lava(), x, y)
            self._lava_positions.append((x, y))

    @property
    def n_lava(self):
        """How many lava cells this episode has.

        Derived rather than stored, so it cannot go stale: a consumer that pins a
        layout assigns to `lava_pos` *after* construction, and the count has to
        follow that list rather than the `n_lava` the env was built with.
        """
        return self._n_lava if self.lava_pos is None else len(self.lava_pos)

    @property
    def lava_positions(self):
        """This episode's lava cells, as a list of (x, y)."""
        return list(self._lava_positions)

    @property
    def goal_reached(self):
        """Whether the square has been reached at any point this episode."""
        return self._goal_paid

    def _on_lava(self):
        cell = self.grid.get(*self.agent_pos)
        return cell is not None and cell.type == "lava"

    def _on_goal(self):
        cell = self.grid.get(*self.agent_pos)
        return cell is not None and cell.type == "goal"

    def _reward(self):
        # Overrides GoalGrid's -1/0: this is the hook GoalGrid.step calls.
        if self._goal_paid or not self._on_goal():
            return 0.0
        return self.goal_reward

    def reset(self, **kwargs):
        self._goal_paid = False
        return super().reset(**kwargs)

    def step(self, action):
        # GoalGrid.step forces terminated=False; recompute it here.
        obs, reward, _, truncated, info = super().step(action)
        terminated = self._on_lava()
        if terminated:
            reward -= self.lava_penalty
        on_goal = self._on_goal()
        info["lava"] = terminated
        info["goal"] = on_goal
        # Marked after super().step(), so the _reward() call inside it saw the
        # pre-step flag: the bonus lands on the step that arrives at the square
        # and never again.
        self._goal_paid = self._goal_paid or on_goal
        return obs, reward, terminated, truncated, info


def make_lava_grid_env(
    size: int = 10,
    agent_start_pos: tuple[int, int] | None = (1, 1),
    agent_start_dir: int = 0,
    max_steps: int = 100,
    goal_pos: tuple[int, int] | None = None,
    n_lava: int = 1,
    lava_penalty: float = 0.0,
    goal_reward: float = 1.0,
    lava_pos: list[tuple[int, int]] | None = None,
    **kwargs,
):
    env = LavaGrid(
        size=size,
        agent_start_pos=agent_start_pos,
        agent_start_dir=agent_start_dir,
        goal_pos=goal_pos,
        max_steps=max_steps,
        n_lava=n_lava,
        lava_penalty=lava_penalty,
        goal_reward=goal_reward,
        lava_pos=lava_pos,
        **kwargs,
    )
    return RGBImgObsWrapper(env)


if __name__ == "__main__":
    # python -m cookie_env.envs.lava_grid
    from cookie_env.utils.play import play

    size = 10
    # rgb_array, not human: the viewer owns the window so it can draw the HUD.
    env = LavaGrid(size=size, max_steps=4 * size, n_lava=3, render_mode="rgb_array")

    play(env)
