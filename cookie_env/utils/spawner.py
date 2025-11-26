import random

VALID_CORNERS = [
    (1, 12),
    (1, 16),
    (27, 12),
    (27, 16)
]

def random_corner():
    return random.choice(VALID_CORNERS)