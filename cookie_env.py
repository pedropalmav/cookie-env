import random
import pygame
import numpy as np
import utils.spawner as spawner
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall, Ball
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from objects import Button, Cookie

class CookieEnv(MiniGridEnv):
    def __init__(
        self,
        height: int = 18,
        width: int = 29,
        agent_start_pos=(14, 14),
        agent_start_dir=0,
        max_steps: int | None = None,
        reward: float = 1.0,
        cookie_spawner=spawner.random_corner,
        **kwargs,
    ):
        self.reward = reward
        self.spawner = cookie_spawner
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.cookie = Cookie()

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * width * height

        super().__init__(
            mission_space=mission_space,
            height=height,
            width=width,
            see_through_walls=True,
            agent_view_size=3,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Get cookies"

    def _gen_grid(self, width, height):
        # TODO: Make a grid generator
        self.grid = Grid(width, height)

        self._fill_with_walls()

        self._generate_room(3, 14)
        self._generate_room(25, 14)
        self._generate_room(14, 3)

        self._generate_hallway(3, 14, 25, 14)
        self._generate_hallway(14, 3, 14, 14)

        self.put_obj(Button("blue", self.place_cookie), 14, 3)

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def _fill_with_walls(self):
        for i in range(self.width):
            for j in range(self.height):
                self.grid.set(i, j, Wall())

    def _generate_room(self, x, y, room_size=5):
        padding = (room_size - 1) // 2
        for i in range(x - padding, x + padding + 1):
            for j in range(y - padding, y + padding + 1):
                self.grid.set(i, j, None)

    def _generate_hallway(self, x1, y1, x2, y2):
        assert x1 == x2 or y1 == y2, "Hallways must be straight"

        if x1 == x2:
            for i in range(abs(y1 - y2) + 1):
                self.grid.set(x1, y1 + i, None)
        else:
            for i in range(abs(x1 - x2) + 1):
                self.grid.set(x1 + i, y1, None)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        obj = self.grid.get(*self.agent_pos)
        if isinstance(obj, Cookie):
            reward += self.reward
            self.grid.set(*self.agent_pos, None)

        return obs, reward, terminated, truncated, info
    
    def place_cookie(self):
        self.remove_cookie()
        pos = self.spawner()
        self.put_obj(self.cookie, *pos)

    def remove_cookie(self):
        if self.cookie.cur_pos is not None:
            self.grid.set(*self.cookie.cur_pos, None)

    def render(self):
        # TODO: Create renderer class
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
                pygame.display.set_caption("Cookie Env")
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
    env = CookieEnv(render_mode="human", screen_size=2048)

    manual_control = ManualControl(env, seed=42)
    manual_control.start()
