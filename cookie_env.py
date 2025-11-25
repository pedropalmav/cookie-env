import pygame
import numpy as np
import utils.spawner as spawner
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from objects import Button, Cookie

IDX_TO_ONEHOT = {
    1: 0,
    2: 1,
    10: 2,
    11: 3,
    12: 4,
}

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
        onehot: bool = False,
        **kwargs,
    ):
        
        self.onehot = onehot
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

        if self.onehot:
            self._init_onehot_obs()

    def _init_onehot_obs(self):
        obs_shape = self.observation_space["image"].shape

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
        objects_types = obs["image"][:, :, 0]
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

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        obj = self.grid.get(*self.agent_pos)
        if isinstance(obj, Cookie):
            reward += self.reward
            self.grid.set(*self.agent_pos, None)

        return obs, reward, terminated, truncated, info
    
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
    env = CookieEnv(render_mode="human", screen_size=2048, onehot=True)

    manual_control = ManualControl(env, seed=42)
    manual_control.start()
