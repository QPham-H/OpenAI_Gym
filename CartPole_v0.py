import gym

env = gym.make('CartPoke-v0')
print("Action space of: ", env.action_space)
print("Observation space of: ", env.observation_space)

for i_episode in range(1000):
  observation = env.reset()
  for t in range(100):
    env.render()
    print(observation)
    action = env. action_space.sample()
    observation, reward, done, info = env.step(action)
    if done
      print("Episode finished after {} timesteps".format(t+1))
      break
 env.close()
