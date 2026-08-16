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

    Lava is placed after the goal and the agent, so it can never spawn on either;
    with the default ``n_lava=1`` a single cell is resampled every reset.

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

    def __init__(self, *args, n_lava: int = 1, lava_penalty: float = 0.0, **kwargs):
        self.n_lava = n_lava
        self.lava_penalty = lava_penalty
        self._lava_positions: list[tuple[int, int]] = []
        super().__init__(*args, **kwargs)

    def _gen_grid(self, width, height):
        # Goal and agent first: place_obj skips occupied cells and the agent's
        # cell, so lava cannot land on either.
        super()._gen_grid(width, height)
        self._lava_positions = [tuple(self.place_obj(Lava())) for _ in range(self.n_lava)]

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
