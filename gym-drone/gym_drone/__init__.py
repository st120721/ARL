from gym.envs.registration import register

register(
    id='Drone2D-v0',
    entry_point='gym_drone.envs:Drone2D',
    # kwargs={
    # }
)
register(
    id='Drone3D-v0',
    entry_point='gym_drone.envs:Drone3D',
    # kwargs={        }
)
