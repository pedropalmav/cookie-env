from minigrid.core.world_object import WorldObj
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.rendering import fill_coords, point_in_circle
from minigrid.core.constants import COLORS, COLOR_TO_IDX

class Button(WorldObj):
    """
    A button that can be pressed by the agent.
    """

    def __init__(self, color: str, action_fn=None):
        super().__init__("ball", color)
        self.action_fn = action_fn

    def toggle(self, env: MiniGridEnv, pos: tuple[int, int]) -> None:
        """
        Method called when the agent presses the button.
        """
        if self.action_fn:
            self.action_fn()

        return True
    
    def encode(self):
        return (11, COLOR_TO_IDX[self.color], 0)

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])