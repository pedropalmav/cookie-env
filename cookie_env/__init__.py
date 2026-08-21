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

register(
    id="TwoRooms-v0",
    entry_point="cookie_env.envs:TwoRooms",
)

register(
    id="GoalEnv-v0",
    entry_point="cookie_env.envs:GoalEnv",
)

# Green-square navigation. The two ids differ only in whether the square is
# pinned (`goal_pos` set) or resampled every episode (`goal_pos=None`).
register(
    id="GoalGrid-v0",
    entry_point="cookie_env.envs:GoalGrid",
    kwargs={"goal_pos": (8, 1), "agent_start_pos": (1, 8)},
)

register(
    id="GoalGrid-random-v0",
    entry_point="cookie_env.envs:GoalGrid",
    kwargs={"goal_pos": None},
)