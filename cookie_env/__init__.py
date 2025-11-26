from gymnasium.envs.registration import register

register(
    id="CookieEnv-v0",
    entry_point="cookie_env.env:CookieEnv",
)

register(
    id="CookieEnv-onehot-v0",
    entry_point="cookie_env.env:CookieEnv",
    kwargs={"onehot": True},
)