"""
General learning Deep NN for OpenAI Gym
Code by Quoc Pham 02.2022
"""

import gym

from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, Flatten, MaxPooling2D, Conv2D
from tensorflow.keras.models import Sequential, Model

# Select OpenAI Gym Atari game to test NN on 
game = 'MountainCar-v0'

EPISODE = 1


env = gym.make(game)

action_space = env.action_space
obs_space = env.observation_space # RGB image of screen (height, width, channels)

img_ht = obs_space[0]
img_wd = obs_space[1]


print(f'This is the action space: {action_space}')
print(f'This is the observation space: {obs_space}')


def build_model():
    input_layer = Input(shape=(img_ht, img_wd,3))
    x = BatchNormalization()(input_layer)
    x = Conv2D(8,3,activation='relu',padding='same')(x)
    x = MaxPooling2D(2)(x)
    x = Conv2D(16,3,activation='relu',padding='same')(x)
    x = MaxPooling2D(2)(x)
    #x = Conv2D(32,3,activation='relu',padding='same')(x)
    #x = MaxPooling2D(2)(x)
    x = Flatten()(x)
    
    #x = Dense(1024, activation='relu')(x)
    #x = Dropout(0.3)(x)
    x = Dense(512,activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(action_space,activation='linear')(x) 
    model = Model(input_layer, out)
    print(model.summary())

    return model
    
