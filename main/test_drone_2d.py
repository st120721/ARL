import gym
import gym_drone

env = gym.make('Drone2D-v0')
env.reset()
for i,a in enumerate([4,4,1,4,4,1,4,1,4,1,1,4,4,4,1,1,2,2,2,2,3,3,3,3,2,2,3,3,3,3,3,2]):
        state = env.state
        # actions=env.get_allowed_actions(state)
        # a = np.random.choice(actions)
        env.step(a)
        env.render()

        print("\nEpisode: ",i )
        print(state)

