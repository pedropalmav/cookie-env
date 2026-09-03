"""Interactive pygame viewer for the GoalGrid family.

MiniGrid's own `ManualControl` prints `step=N, reward=X` to the console and
resets the moment an episode ends. That is a bad fit for these envs: the reward
is sparse (`LavaGoalGrid` pays 0 on almost every step), reaching the green
square does not end the episode, and the instant reset hides *how* the episode
ended. This viewer draws the state that actually matters next to the grid and
waits for a keypress before starting the next episode.

It reads everything through `env.unwrapped` and probes optional attributes, so
it works for `GoalGrid`, `LavaGrid` and `LavaGoalGrid` alike, wrapped or not.
The env must be built with `render_mode="rgb_array"` — this module owns the
window, rather than letting MiniGrid open its own.
"""

import numpy as np
import pygame

HUD_HEIGHT = 128
PAD = 16

PALETTE = {
    "bg": (18, 18, 20),
    "text": (232, 232, 236),
    "dim": (138, 138, 148),
    "good": (108, 208, 128),
    "bad": (234, 98, 98),
    "warn": (234, 198, 108),
}

# MiniGrid's Actions enum, by the key that triggers it.
KEY_TO_ACTION = {
    pygame.K_LEFT: 0,
    pygame.K_RIGHT: 1,
    pygame.K_UP: 2,
    pygame.K_TAB: 3,
    pygame.K_PAGEUP: 3,
    pygame.K_LSHIFT: 4,
    pygame.K_PAGEDOWN: 4,
    pygame.K_SPACE: 5,
    pygame.K_RETURN: 6,
}


class Player:
    """Blocking pygame loop: arrows to move, R to reset, Esc to quit."""

    def __init__(self, env, seed: int | None = None, window_px: int = 720, fps: int = 30):
        self.env = env
        self.seed = seed
        self.window_px = window_px
        self.fps = fps

        self.total = 0.0
        self.last_reward = 0.0
        self.outcome = None  # None while alive, else "lava" / "time"
        self.episodes = 0
        self.successes = 0

    @property
    def core(self):
        return self.env.unwrapped

    def start(self):
        pygame.init()
        pygame.display.set_caption(f"cookie_env — {type(self.core).__name__}")
        self.font = pygame.font.Font(None, 26)
        self.small = pygame.font.Font(None, 21)
        self.clock = pygame.time.Clock()

        self._reset()
        frame = self._frame()
        height = round(self.window_px * frame.shape[0] / frame.shape[1])
        self.screen = pygame.display.set_mode((self.window_px, height + HUD_HEIGHT))

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._on_key(event.key)
            self._draw()
            self.clock.tick(self.fps)

        self.env.close()
        pygame.quit()

    # ── input ────────────────────────────────────────────────────────────

    def _on_key(self, key):
        if key == pygame.K_ESCAPE:
            return False
        if key in (pygame.K_r, pygame.K_BACKSPACE):
            self._reset()
            return True
        # Once the episode is over the grid is frozen: only reset and quit act.
        if self.outcome is None and key in KEY_TO_ACTION:
            self._step(KEY_TO_ACTION[key])
        return True

    def _step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        self.last_reward = float(reward)
        self.total += float(reward)
        if terminated:
            self.outcome = "lava" if info.get("lava") else "terminated"
        elif truncated:
            self.outcome = "time"
        if self.outcome is not None:
            self.episodes += 1
            if self._goal_reached():
                self.successes += 1

    def _reset(self):
        self.env.reset(seed=self.seed)
        self.total = 0.0
        self.last_reward = 0.0
        self.outcome = None

    # ── env probing ──────────────────────────────────────────────────────

    def _goal_reached(self):
        """True once the square has been reached this episode.

        `LavaGoalGrid` tracks this across the episode because arriving is not
        terminal; the others only ever know where the agent is standing now.
        """
        reached = getattr(self.core, "goal_reached", None)
        if reached is not None:
            return bool(reached)
        cell = self.core.grid.get(*self.core.agent_pos)
        return cell is not None and cell.type == "goal"

    def _frame(self):
        return self.env.render()

    # ── drawing ──────────────────────────────────────────────────────────

    def _draw(self):
        self.screen.fill(PALETTE["bg"])

        surface = pygame.surfarray.make_surface(np.transpose(self._frame(), (1, 0, 2)))
        grid_h = self.screen.get_height() - HUD_HEIGHT
        self.screen.blit(pygame.transform.scale(surface, (self.window_px, grid_h)), (0, 0))

        self._draw_hud(grid_h)
        pygame.display.flip()

    def _draw_hud(self, top):
        core = self.core
        reached = self._goal_reached()

        if self.outcome == "lava":
            status, color = "DIED IN LAVA — press R", PALETTE["bad"]
        elif self.outcome is not None:
            status, color = "TIME LIMIT — press R", PALETTE["warn"]
        elif reached:
            status, color = "goal reached", PALETTE["good"]
        else:
            status, color = "searching", PALETTE["dim"]

        # place_obj hands back np.int64s; format them as plain ints.
        gx, gy = (int(v) for v in core.goal_position)
        goal_line = f"goal ({gx},{gy})"
        if reached:
            goal_line += " — collected"

        left = [
            (f"step {core.step_count}/{core.max_steps}", PALETTE["text"]),
            (f"return {self.total:+.2f}    last {self.last_reward:+.2f}", PALETTE["text"]),
            (goal_line, PALETTE["good"] if reached else PALETTE["dim"]),
        ]
        right = [
            (status, color),
            (f"episodes {self.episodes}    reached {self.successes}", PALETTE["dim"]),
        ]
        if hasattr(core, "lava_positions"):
            right.append((f"lava cells {len(core.lava_positions)}", PALETTE["dim"]))

        y = top + PAD
        for text, tint in left:
            self.screen.blit(self.font.render(text, True, tint), (PAD, y))
            y += 27

        y = top + PAD
        for text, tint in right:
            label = self.small.render(text, True, tint)
            self.screen.blit(label, (self.screen.get_width() - PAD - label.get_width(), y))
            y += 24

        hint = "arrows: turn / forward     R: reset     Esc: quit"
        self.screen.blit(
            self.small.render(hint, True, PALETTE["dim"]),
            (PAD, self.screen.get_height() - PAD - 14),
        )


def play(env, seed: int | None = None, window_px: int = 720):
    """Open the viewer on `env` (built with render_mode="rgb_array")."""
    Player(env, seed=seed, window_px=window_px).start()
