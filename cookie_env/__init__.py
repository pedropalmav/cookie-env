from gymnasium.envs.registration import register
from cookie_env.utils.spawner import deterministic_corner

register(
    id="CookieEnv-v0",
    entry_point="cookie_env.env:CookieEnv",
)

register(
    id="CookieEnv-onehot-v0",
    entry_point="cookie_env.env:CookieEnv",
    kwargs={"onehot": True},
)

register(
    id="CookieEnv-deterministic-v0",
    entry_point="cookie_env.env:CookieEnv",
    kwargs={"cookie_spawner": deterministic_corner},
)

register(
    id="CookieEnv-norespawn-v0",
    entry_point="cookie_env.env:CookieEnv",
    kwargs={"respawn": False},
)
