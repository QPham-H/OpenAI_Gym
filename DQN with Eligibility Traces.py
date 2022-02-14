"""
General learning Deep NN for OpenAI Gym
Code by Quoc Pham 02.2022
"""

# Install dependencies via
# pip install gym[atari,accept-rom-license]==0.21.0
# pip install pyglet

import gym
import random
import numpy as np

from tensorflow.keras.layers import BatchNormalization, concatenate, Dense, Dropout, Input, Flatten, MaxPooling2D, Conv2D, Rescaling
from tensorflow.keras.models import Sequential, Model

from collections import deque

# Select OpenAI Gym Atari game to test NN on
game = 'AirRaid-v0'

# Constants and Variables
EPISODES = 10
STEPS = 100
EPSILON = 0.2
ALPHA = 0.1 # Learning rate
DISCOUNT = 0.9
LAMBDA = 0.9 # For Eligibility trace
BATCH = 32
MEMORY = BATCH * 10
EPOCHS = 1
display = True

# Begin creating the environment 
env = gym.make(game) #, render_mode='human') # Using `render_mode` provides access to proper scaling, audio support, and proper framerates

action_space = env.action_space.n # Number of actions
obs_space = env.observation_space.shape # RGB image of screen (height, width, channels)

img_ht = obs_space[0]
img_wd = obs_space[1]


print(f'This is the number of actions: {action_space}')
print(f'This is the observation image: {obs_space}')


input_layer = Input(shape=(img_ht, img_wd,3),name='image')
x = Rescaling(1/255)(input_layer)
x = BatchNormalization()(x)
x = Conv2D(8,3,activation='relu',padding='same')(x)
x = MaxPooling2D(2)(x)
x = Conv2D(16,3,activation='relu',padding='same')(x)
x = MaxPooling2D(2)(x)
x = Conv2D(8,3,activation='relu',padding='same')(x)
x = MaxPooling2D(2)(x)
#x = Conv2D(32,3,activation='relu',padding='same')(x)
#x = MaxPooling2D(2)(x)
cnn = Flatten()(x)

action_layer = Input(shape=(action_space),name='action')

#x = Dense(1024, activation='relu')(x)
#x = Dropout(0.3)(x)
##x = Dense(512,activation='relu')(cnn)
##x = Dropout(0.3)(x)

x = concatenate([cnn, action_layer])
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(1,activation='linear')(x) 

q_model = Model([input_layer,action_layer], out)
print(q_model.summary())


replay_memory = deque()
q_dict = dict()

def action_selection(state):
    state = np.expand_dims(state,axis=0)    # Need this to use predict on one sample
    
    q_values = np.zeros(action_space)
    for a in range(action_space):
        one_hot_action = np.zeros(action_space)
        one_hot_action[a] = 1

        one_hot_action = np.expand_dims(one_hot_action,axis=0)
        
##        print(f'This is state shape: {state.shape}')
##        print(f'This is action shape: {one_hot_action.shape}')
        
        q_values[a] = q_model.predict([state, one_hot_action])
        
    if random.random() <= EPSILON:
        action = random.randint(0, action_space - 1)
    else:
        action = np.argmax(q_values)
        
    return action, q_values[action]

def propagate_delta(curr_q_value, next_state, reward, t):
    next_state = np.expand_dims(next_state,axis=0)

    next_q_values = np.zeros(action_space)
    for a in range(action_space):
        one_hot_action = np.zeros(action_space)
        one_hot_action[a] = 1
        
        one_hot_action = np.expand_dims(one_hot_action,axis=0)
        
        next_q_values[a] = q_model.predict([next_state, one_hot_action])

    delta = reward + DISCOUNT * np.max(next_q_values) - curr_q_value

    # Eligibility traces are automatically 1 for most recent time step if there's a reward
    # Propagate only until traces decay significantly enough
    
    for i in range(len(replay_memory)):
        n = t - replay_memory[i][5] # where n is the time steps away from current time
        z = (LAMBDA * DISCOUNT) ** n
        
        if z <= 0.1:
            break

        replay_memory[i][2] += ALPHA * delta * z
    

def train_model():
    sessions = floor(len(replay_memory)/BATCH)

    model.compile(
    loss='mse',
    optimizer='adam'
    )

    one_hot_actions = np.zeros(BATCH, action_space)
    for n in range(sessions):
        batch = random.sample(replay_memory, BATCH)

        states = [data[0] for data in batch]
        actions = [data[1] for data in batch]
        one_hot_actions[np.arange(BATCH),actions] = 1
        print('This is one_hot_actions')
        print(one_hot_actions)
        q_values = [data[2] for data in batch]

        history = model.fit(
            [states, actions],
            q_values,
            batch_size = BATCH,
            epochs = EPOCHS
            )


for episode in range(EPISODES):
    print(f'Start Episode {episode}')
    done = False
    state = env.reset() # Returns an observation (RGB image)

    episode_reward = 0
    for t in range(STEPS):
        if display:
            env.render() # Render the image

        action, q_value = action_selection(state)
        next_state, reward, done, info = env.step(action)

        replay_memory.appendleft((state, action, q_value, next_state, done, t))

        if len(replay_memory) > MEMORY: # To control memory size from exploding
            replay_memory.popright()

        if reward != 0: # If there's a reward, give credit to recent states based on eligibility traces
            propagate_delta(q_value, next_state, reward, t)
            episode_reward += reward

        if done:
            print(f'End of episode {episode} at {t} time steps')
            break

        state = next_state
        
    train_model() # Update model between successive episodes
    print(f'Reward for Episode {episode}: {episode_reward}')
    

env.close()
