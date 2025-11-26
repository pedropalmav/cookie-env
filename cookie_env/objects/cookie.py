from minigrid.core.world_object import WorldObj
from minigrid.utils.rendering import fill_coords, point_in_circle
from minigrid.core.constants import COLORS, COLOR_TO_IDX
import numpy as np
import pygame
import os

class Cookie(WorldObj):
    """
    A cookie that can be collected by the agent.
    """
    
    _cached_image = None
    _image_loaded = False

    def __init__(self):
        self.color = "yellow"
        super().__init__("ball", self.color)
        
        if not Cookie._image_loaded:
            Cookie._load_cookie_image()

    @classmethod
    def _load_cookie_image(cls):
        try:
            image_path = cls._get_image_path()
            
            if os.path.exists(image_path):
                pygame.init()
                
                cls._cached_image = pygame.image.load(image_path)
                cls._image_loaded = True
        except Exception as e:
            cls._image_loaded = True

    @classmethod
    def _get_image_path(cls):
        current_dir = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(current_dir, "assets", "cookie.png")

    def can_overlap(self):
        return True
    
    def encode(self):
        return (12, COLOR_TO_IDX[self.color], 0)

    def render(self, img):
        if Cookie._cached_image is not None:
            target_height, target_width = img.shape[:2]
            
            new_width, new_height, cookie_surface = self.scale((target_height, target_width))
            
            cookie_array = pygame.surfarray.array3d(cookie_surface)
            cookie_array = np.transpose(cookie_array, (1, 0, 2))
            
            offset_x = (target_width - new_width) // 2
            offset_y = (target_height - new_height) // 2
            
            end_y = min(offset_y + new_height, target_height)
            end_x = min(offset_x + new_width, target_width)
            img[offset_y:end_y, offset_x:end_x] = cookie_array[:end_y-offset_y, :end_x-offset_x]

        else:
            fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

    def scale(self, target_shape):
        original_width, original_height = Cookie._cached_image.get_size()

        margin = 0.05
        available_width = target_shape[1] * (1 - 2 * margin)
        available_height = target_shape[0] * (1 - 2 * margin)
        
        scale_x = available_width / original_width
        scale_y = available_height / original_height
        scale = min(scale_x, scale_y)
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        return new_width, new_height, pygame.transform.scale(Cookie._cached_image, (new_width, new_height))