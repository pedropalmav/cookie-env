import random

VALID_CORNERS = [(1, 12), (1, 16), (27, 12), (27, 16)]


def random_corner():
    return random.choice(VALID_CORNERS)


def deterministic_corner(corner_id=0):
    return VALID_CORNERS[corner_id]
