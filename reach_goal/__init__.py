from gym.envs.registration import register
register(
    id='PointMass-v0',
    entry_point='reach_goal.envs.point_mass_env:PointMassEnv'
)