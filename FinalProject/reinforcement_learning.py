import gym
import gym_drone
import pygame
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import numpy as np
import random
import matplotlib.pyplot as plt
import time
plt.close('all')

# Initialize action space
action_map = {'Up': 0, 'Right': 1, 'Down': 2, 'Left': 3}
num_actions = len(action_map)

# Set number of steps to run through
steps = 80
count = 0
# If run is flse, will stop execution
run = True
# Use for testing... remove later
close_map = run

# Initialize Q-Learning params
q_table = np.zeros([100,4])

#discount factor to scale future rewards gamma
gamma = 0.99

#learning rate alpha
alpha = 0.001

#number of episodes to train
total_episodes = 1500

# Exploration parameters
epsilon = 0.9            # Exploration rate
max_epsilon = 0.8           # Exploration probability at start
min_epsilon = 0.2           # Minimum exploration probability 
decay_rate = 0.005            # Exponential decay rate for exploration pro

# Max number of actions before cutting an epsiode to begin next episode
max_actions = 250

# Get maximum function
def maximum(list_of_values):
    maximum = list_of_values[0]
    for value in list_of_values[1:]:
        check = value > maximum
        if(True in check):
            maximum = value
    return np.argmax(maximum)

#update Q_table function
def updateQTable(existing_table, new_state_observations, last_state, last_action):
    #find the maximum Q value of the new state we reached among all the possible actions
    new_state = new_state_observations[0]
    reward_of_last_action = new_state_observations[1]
    maxQ = maximum(existing_table[new_state, :])
    existing_table[last_state, last_action] = (1-alpha)*existing_table[last_state, last_action] + alpha*(reward_of_last_action + gamma*maxQ)
    
    return existing_table

# Load the environment
env = gym.make('drone-v0')
# Initialize parameters
env.reset()

# Main loop
while run:
    keys = pygame.key.get_pressed()
    pygame.time.delay(100)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = True
    #action = random.randint(0,3)
    # New action for agent
    #env.step(action)
    # Update environment
    env.render()
    
    # Press q to Quit 
    if keys[ord('q')]:
        run = False
        close_map = True
    # Stops running if reached max steps
    if count > steps:
        run = False
        close_map = True
    # Update counter
    count += 1 
    
# Closes the environment
if close_map:
    env.close()
'''
# Create neural network
def net():
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    print(model.summary())
    
    return model

model = net()
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=5000, window_length=1)
# dqn = DQNAgent() #continue research and modify gym environment
'''

fix = {4: 0, 5: 1, 6:2, 7: 3}
action_plot = []
random_actions = 0
q_learning_actions = 0
reward_plot = []
ep_step = 50
episode_plot = np.arange(ep_step, total_episodes+1, step=ep_step)
rewards_ep = 0
now = time.time()
end = time.time()
# Train the Agent to avoid obstacles and reach the finish line
for episode in range(total_episodes):
    if (episode+1) % ep_step == 0:
        print('Episode: ', episode+1)
        print('Time:',str(end-now)[0:4],'seconds')
        print('Rewards:', rewards_ep/100)
        print('Epsilon:', epsilon,'\n')
        now = time.time()
        reward_plot.append(rewards_ep/100)
        rewards_ep = 0
        
    env.reset()
    
    #start the episode
    episode_end = False
    current_state = 0
    while_count  = 0
    action_count = 0
    while(not episode_end):
        while_count += 1
        # Take a random number from 0 to 1 for Epsilon Greedy Policy
        random_number = random.uniform(0,1)
        if(random_number <= epsilon):
            # Perform random action
            action = random.randint(0,num_actions-1)
            observations = env.step(action)
            q_table = updateQTable(q_table, observations, current_state, action)
            current_state = observations[0]
            episode_end = observations[2]
            random_actions += 1
        else:
            # Use Q-Table
            action = np.argmax(q_table[current_state, :])
            if action > 3:
                action = fix[action]
            observations = env.step(action)
            q_table = updateQTable(q_table, observations, current_state, action)
            current_state = observations[0]
            episode_end = observations[2]
            q_learning_actions += 1
        rewards_ep += observations[1]
        
        if while_count >= max_actions:
            episode_end = True
            
        action_count += 1
    # Store num action results
    action_plot.append(action_count)
    
    # Update decaying epsilon greedy policy
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)     
    
    # Record time to finish episode
    end = time.time()
env.close()

# Results
print('Completed Training')
print('Percentage random actions: '+str(random_actions/(random_actions+q_learning_actions)*100)[0:4] + '%')
print('Percentage q-learning actions: '+str(q_learning_actions/(random_actions+q_learning_actions)*100)[0:4] + '%')
print('Final epsilon: ', epsilon)

# Modify to take mean of num actions for every episode step count
a = []
for i in range(len(reward_plot)):
    a.append(np.mean(action_plot[i*ep_step:i*ep_step+ep_step]))
    
plt.figure()
plt.plot(episode_plot, a)
plt.title('Q-Learning Model Actions vs. Episodes')
plt.xlabel('Episodes')
plt.ylabel('Number of Actions')

plt.figure()
plt.plot(episode_plot, reward_plot)
plt.title('Q-Learning Model Reward vs Episodes')
plt.xlabel('Episodes')
plt.ylabel('Rewards')

plt.show()