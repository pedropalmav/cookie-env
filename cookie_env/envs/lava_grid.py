from minigrid.core.world_object import Lava
from minigrid.wrappers import RGBImgObsWrapper

from cookie_env.envs.goal_grid import GoalGrid


class LavaGrid(GoalGrid):
    """`GoalGrid` plus lava cells that end the episode on contact.

    This is the episodic counterpart of `GoalGrid`: reaching the green square
    still does *not* terminate (it only stops the -1 per step), but stepping into
    lava sets ``terminated=True``. Death is the only terminal, which keeps the
    return comparable to `GoalGrid` for as long as the agent stays alive, and
    matches environments like Crafter where there is no single goal cell but
    dying does end the episode.

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

    ``lava_penalty`` is subtracted on the terminating step and defaults to 0, so
    dying simply ends the episode.

    Note on the reward scale in general, independent of this parameter: with a
    negative per-step reward, terminating is worth more than surviving, because
    ending the episode stops the accumulation of -1s. Measured here with a random
    policy under a 0.997 discount, dying scored -136 against -309 for running to
    the time limit. Any agent that can perceive termination has an incentive to
    seek the lava. Whoever consumes this env has to answer that — either through
    ``lava_penalty`` or with a reward scheme that is never negative.

    ``lava_penalty`` only applies if the consumer actually uses the reward this
    env returns. Some do not: her-dream overwrites ``reward`` with a
    goal-conditioned latent reward before the transition reaches its buffer, so
    this parameter has no effect there and the penalty has to live on that side.
    """

    def __init__(
        self,
        *args,
        n_lava: int = 1,
        lava_penalty: float = 0.0,
        lava_pos: list[tuple[int, int]] | None = None,
        **kwargs,
    ):
        self.lava_pos = None if lava_pos is None else [(int(x), int(y)) for x, y in lava_pos]
        self._n_lava = n_lava
        self.lava_penalty = lava_penalty
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

    def _on_lava(self):
        cell = self.grid.get(*self.agent_pos)
        return cell is not None and cell.type == "lava"

    def step(self, action):
        # GoalGrid.step forces terminated=False; recompute it here.
        obs, reward, _, truncated, info = super().step(action)
        terminated = self._on_lava()
        if terminated:
            reward -= self.lava_penalty
        info["lava"] = terminated
        return obs, reward, terminated, truncated, info


def make_lava_grid_env(
    size: int = 10,
    agent_start_pos: tuple[int, int] | None = (1, 1),
    agent_start_dir: int = 0,
    max_steps: int = 100,
    goal_pos: tuple[int, int] | None = None,
    n_lava: int = 1,
    lava_penalty: float = 0.0,
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
        lava_pos=lava_pos,
        **kwargs,
    )
    return RGBImgObsWrapper(env)


if __name__ == "__main__":
    # python -m cookie_env.envs.lava_grid
    from minigrid.manual_control import ManualControl

    size = 10
    env = make_lava_grid_env(size=size, max_steps=2 * size, render_mode="human")

    manual_control = ManualControl(env)
    manual_control.start()
