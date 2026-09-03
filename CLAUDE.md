# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`cookie_env` is a **library of MiniGrid environments** consumed by external RL training code
(notably a DreamerV3-style project referred to in docstrings as *her-dream*). It is not an
application and has no training loop of its own. The public surface is: the Gymnasium ids
registered in `cookie_env/__init__.py`, the classes exported from `cookie_env/envs/__init__.py`,
and the `make_*_env` helpers.

Because it is a library, `pyproject.toml` deliberately declares **lower-bound-only** dependencies
on just what `cookie_env` imports (gymnasium, minigrid, numpy, pygame). Do not re-add `==` pins or
dev-only packages there — pinned dev versions live in `requirements.txt` / `uv.lock`.

## Commands

```bash
uv sync                                   # install (incl. dev group: pytest)
uv run pytest                             # all tests (testpaths = tests/)
uv run pytest tests/test_lava_grid.py     # one file
uv run pytest tests/test_goal_grid.py -k test_gym_make   # one test
uv run pytest -k "fixed"                  # one variant across the parametrized fixtures

uv export --frozen --no-hashes --no-emit-project --format requirements-txt -o requirements.txt
```

Every env module is runnable for manual play via `ManualControl` (opens a pygame window):

```bash
uv run python -m cookie_env.envs.goal_grid
uv run python -m cookie_env.envs.lava_grid
uv run python -m cookie_env.envs.lava_goal_grid   # uses the HUD viewer below
uv run python -m cookie_env.envs.random_goal
```

`cookie_env/utils/play.py` is a pygame viewer for the `GoalGrid` family that replaces
`ManualControl`: it draws step/return/goal/outcome next to the grid and freezes the final frame
until R, instead of printing to the console and resetting instantly. It matters here because these
rewards are sparse and reaching the square is not terminal, so the console output alone does not
say what happened. Build the env with `render_mode="rgb_array"` — the viewer owns the window — and
call `play(env)`. It probes optional attributes (`goal_reached`, `lava_positions`) so it works for
all three envs in the chain.

Key bindings (from README): Left/Right = turn, Up = forward, Tab = pick up, Shift/PageDown = drop,
Space = toggle, Enter = done. `GoalEnv` adds action `7` (idle) bound to Space via a subclassed
`HERManualControl`.

There is no linter or formatter configured.

## Two env families

**Cookie/button family** — `ThreeRooms`, `TwoRooms`, `Corridor`, `CornerEnv`. Walled layouts where
a `Button` (`cookie_env/objects/button.py`) calls back into the env on `toggle()` to spawn a
`Cookie` at a position chosen by a *spawner function* (`cookie_env/utils/spawner.py`, passed as the
`cookie_spawner` kwarg). Positive reward on collecting the cookie. These four share a large amount
of copy-pasted code (`_init_onehot_obs`, `_get_onehot_obs`, `render`, `_generate_hallway`, the
`IDX_TO_ONEHOT` table) — a change to one usually needs mirroring in the others, and the `render`
override exists only to size the pygame window for non-square grids (the README's open TODOs are a
renderer class and a grid-generator class).

**Goal-navigation family** — `GoalEnv` (`random_goal.py`), then the `GoalGrid` line. Empty rooms
designed to be driven by an external world model. This is where active work happens.

    GoalGrid            -1/step, 0 on the green square, never terminates
      └─ LavaGrid       + lava; death is the only terminal
           └─ LavaGoalGrid   0/step, +1 once on the square, 0 on death

All three share `make_*_env` helpers that wrap the env in `RGBImgObsWrapper` (pixel observations)
and register their variants as ids. Prefer extending this chain over cloning it.

## Conventions that matter in the goal-navigation family

- **`goal_pos=None` means "resample each episode"**, mirroring MiniGrid's `agent_start_pos=None`.
  `goal_pos` is the *configured* value (may be `None`); `_goal_pos` is the concrete cell for the
  current episode; consumers read it through the `goal_position` property. Consumers that pin a
  layout assign to `goal_pos` and call `reset()`.
- **Reaching the goal does not terminate.** `GoalGrid.step` forces `terminated=False` and returns
  `0` instead of `-1` on the goal cell; episode end is signalled by `truncated` (time limit).
  Return is therefore a step-count proxy, not a success flag. `LavaGrid` re-derives `terminated`
  after calling `super().step()` — lava contact is the only terminal.
- **Negative per-step reward makes termination attractive.** Read `LavaGrid`'s docstring before
  tuning rewards: with -1/step, dying scores better than surviving to the time limit
  (-136 vs -309 measured with a random policy at γ=0.997). `lava_penalty` is the knob, but it is
  inert for consumers that overwrite `reward` with their own goal-conditioned signal.
  `LavaGoalGrid` is the non-negative answer to this — 0/step, +1 on the square, 0 on death — and
  inverts the comparison (0.31 dying vs 0.47 surviving under the same measurement).
- **A positive goal reward must be paid once per episode.** Since reaching the square is not
  terminal, `LavaGoalGrid` gates the bonus behind `_goal_paid` so a parked agent cannot farm it;
  the flag is set in `step` *after* `super().step()` so `_reward()` still sees the pre-step value.
  Episode return is then a success flag in `{0, goal_reward}`, not a step-count proxy — and the
  reward is sparse, so exploration has to come from the consumer.
- **Ordering in `_gen_grid` is load-bearing.** `LavaGrid` calls `super()._gen_grid()` first so goal
  and agent are placed before lava; `place_obj` then cannot land lava on either.
- **Missions carry state as text.** `GoalGrid` rewrites `obs["mission"]` on every `reset`/`step` to
  `"agent at (x,y) facing <dir>. goal at (x,y)"`. `random_mission()` samples strings from the same
  distribution for consumers that need synthetic missions — keep the two formats in sync.
- **Register new variants as ids** in `cookie_env/__init__.py` with kwargs, rather than adding a
  subclass, when the only difference is configuration.

## Observation formats

`onehot=True` (cookie family, `GoalEnv`) replaces the `(H,W,3)` MiniGrid image with a `(H,W,5)`
uint8 one-hot over `IDX_TO_ONEHOT` — MiniGrid type ids `1` (empty), `2` (wall), `10` (agent),
plus `11` (button) and `12` (cookie).
`Button` and `Cookie` override `encode()` to claim the otherwise-unused type ids 11 and 12; the
agent's own cell is stamped in manually at `[view_size//2, view_size-1]`.

`GoalEnv` instead appends a `goal` key: a double one-hot (`stoch_rows ⊕ stoch_classes`) matching a
DreamerV3 latent, or just the class one-hot when `fixed_row=True`. It has no visible `Goal` object —
the target lives entirely in latent space and the consumer assigns reward.

## Tests

`tests/` covers only `GoalGrid` and `LavaGrid`. They are organised as behaviour classes
(`TestInit`, `TestGoalPlacement`, …) with a `variant` fixture parametrized over `"fixed"` /
`"random"` so shared behaviour is asserted for both goal modes. Tests import `cookie_env` for the
`register()` side effect and exercise both direct construction and `gym.make(...)`.

Docstrings and comments in this repo explain *why* a design choice was made (see `goal_grid.py`,
`lava_grid.py`, `pyproject.toml`) rather than restating the code. Match that when adding code.
Commit messages use `feature:` / `fix:` prefixes.
