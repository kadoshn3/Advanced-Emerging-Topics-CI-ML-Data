from gym.envs.registration import register

register(
    id='neeve-v0',
    entry_point='gym_neeve.envs:NeeveEnv',
)