import pygame
import numpy as np
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

# Reusa el mapeo que usas en Corridor si quieres onehot
IDX_TO_ONEHOT = {
    1: 0,
    2: 1,
    10: 2,
    11: 3,
    12: 4,
}


class CornerEnv(MiniGridEnv):
    """
    Corner environment:
    - square grid size x size
    - a goal in a corner (corner_goal_pos) with high reward
    - optional onehot image observation
    """

    def __init__(
        self,
        size: int = 7,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        reward_corner: float = 1.0,
        onehot: bool = False,
        render_mode: str | None = None,
        **kwargs,
    ):
        self.size = size
        self.onehot = onehot
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.reward_corner = reward_corner

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 10_000

        super().__init__(
            mission_space=mission_space,
            height=size,
            width=size,
            see_through_walls=True,
            agent_view_size=3,
            max_steps=max_steps,
            render_mode=render_mode,
            **kwargs,
        )

        # Replace image observation to onehot channels (H,W,5) if requested
        if self.onehot:
            self._init_onehot_obs()

        # placeholders (set in _gen_grid)
        self.corner_goal_pos = None

    def _init_onehot_obs(self):
        obs_shape = self.observation_space["image"].shape  # (H, W, C)
        new_image_space = spaces.Box(
            low=0, high=1, shape=(obs_shape[0], obs_shape[1], 5), dtype="uint8"
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def gen_obs(self):
        obs = super().gen_obs()
        if self.onehot:
            return self._get_onehot_obs(obs)
        return obs

    def _get_onehot_obs(self, obs):
        # obs["image"] is H x W x C, where channel 0 is object type
        objects_types = obs["image"][:, :, 0].copy()

        # mark the agent cell in the view just like Corridor does (optional)
        # center row, rightmost col in agent view: (agent_view_size//2, agent_view_size-1)
        objects_types[self.agent_view_size // 2, self.agent_view_size - 1] = 10

        onehot_image = np.zeros(
            (obs["image"].shape[0], obs["image"].shape[1], 5), dtype="uint8"
        )

        for i in range(objects_types.shape[0]):
            for j in range(objects_types.shape[1]):
                obj_type = objects_types[i, j]
                if obj_type in IDX_TO_ONEHOT:
                    onehot_idx = IDX_TO_ONEHOT[obj_type]
                    onehot_image[i, j, onehot_idx] = 1

        obs["image"] = onehot_image
        return obs

    @staticmethod
    def _gen_mission():
        return "Go to the corner"

    def _gen_grid(self, width, height):
        # Create empty grid and surrounding walls
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # corner goal (bottom-right-ish inside wall)
        self.corner_goal_pos = (width - 2, height - 2)

        # Place the goals
        self.put_obj(Goal(), *self.corner_goal_pos)

        # Place the agent (fixed start pos or random)
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        agent_pos_tuple = tuple(self.agent_pos)

        if terminated:
            if agent_pos_tuple == self.corner_goal_pos:
                reward = self.reward_corner
            
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        # no special cleanup needed, but keep API parity with Corridor
        return super().reset(seed=seed, options=options)

    def render(self):
        # reuse MiniGrid rendering pipeline
        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

        if self.render_mode == "human":
            img = np.transpose(img, axes=(1, 0, 2))
            if self.render_size is None:
                self.render_size = img.shape[:2]
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.screen_size, self.screen_size * self.height // self.width)
                )
                pygame.display.set_caption("Corner Env")
            if self.clock is None:
                self.clock = pygame.time.Clock()
            surf = pygame.surfarray.make_surface(img)

            window_size = self.window.get_size()
            surf = pygame.transform.scale(surf, window_size)

            self.window.blit(surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return img


if __name__ == "__main__":
    env = CornerEnv(size=9, render_mode="human", onehot=True)
    manual_control = ManualControl(env, seed=42)
    manual_control.start()
