from gymnasium.envs.registration import register
from cookie_env.utils.spawner import deterministic_corner

register(
    id="CookieEnv-v0",
    entry_point="cookie_env.envs:ThreeRooms",
)

register(
    id="CookieEnv-onehot-v0",
    entry_point="cookie_env.envs:ThreeRooms",
    kwargs={"onehot": True},
)

register(
    id="CookieEnv-deterministic-v0",
    entry_point="cookie_env.envs:ThreeRooms",
    kwargs={"cookie_spawner": deterministic_corner},
)

register(
    id="CookieEnv-norespawn-v0",
    entry_point="cookie_env.envs:ThreeRooms",
    kwargs={"respawn": False},
)

register(
    id="Corridor-v0",
    entry_point="cookie_env.envs:Corridor",
)

register(
    id="CornerEnv-v0",
    entry_point="cookie_env.envs:CornerEnv",
)