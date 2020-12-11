import gym
import gym_neeve
import pygame
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
import matplotlib.pyplot as plt

plt.close('all')

# Initialize action space
action_map = {'Up': 0, 'Right': 1, 'Down': 2, 'Left': 3}
num_actions = len(action_map)

# Set number of steps to run through
steps = 20
count = 0
# If run is flse, will stop execution
run = False
# Use for testing... remove later
close_map = False


# Initialize Q-Learning params
q_table = np.zeros([150,4])

#discount factor to scale future rewards gamma
gamma = 0.95

#learning rate alpha
alpha = 0.8

#number of episodes to train
total_episodes = 1000

# Exploration parameters
epsilon = 5                 # Exploration rate
max_epsilon = 0.9            # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.01             # Exponential decay rate for exploration pro

#maximum function
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
env = gym.make('neeve-v0')
# Initialize parameters
env.reset()

# Main loop
while run:
    keys = pygame.key.get_pressed()
    pygame.time.delay(100)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = True
    if count < 50:
        action = action_map['Right']
    else:
        action = action_map['Left']
    # New action for agent
    env.step(action)
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
fix = {4: 0, 5: 1, 6:2, 7: 3}
action_plot = []
random_actions = 0
q_learning_actions = 0
for episode in range(total_episodes):
    if (episode+1) % 50 == 0:
        print('Episode: ', episode+1)
    env.reset()
    
    #start the episode
    episode_end = False
    current_state = 0
    while_count  = 0
    action_count = 0
    while(not episode_end):
        while_count += 1
        #finding the action to take next from present state
        #generate a random number in [0,1) 
        # if random_number < epsilon => take random action
        # else take an action given by max Q value at that state
        random_number = random.uniform(0,1)
        if(random_number <= epsilon):
            #take random step
            action = random.randint(0,num_actions-1)
            observations = env.step(action)
            q_table = updateQTable(q_table, observations, current_state, action)
            current_state = observations[0]
            episode_end = observations[2]
            random_actions += 1
        else:
            #take step told by Q table
            '''if two values are same this will always pick the action corresponding to the first value'''
            action = np.argmax(q_table[current_state, :])
            if action > 3:
                action = fix[action]
            observations = env.step(action)
            q_table = updateQTable(q_table, observations, current_state, action)
            current_state = observations[0]
            episode_end = observations[2]
            q_learning_actions += 1
        
        if while_count >= 100:
            episode_end = True
        action_count += 1

    action_plot.append(action_count)
    #exploitation vs exploration epsilon-greedy technique
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
env.close()
# Results
print('Completed Training')
print('Percentage random actions: '+str(random_actions/(random_actions+q_learning_actions)*100)[0:4] + '%')
print('Percentage q-learning actions: '+str(q_learning_actions/(random_actions+q_learning_actions)*100)[0:4] + '%')
print('Final epsilon: ', epsilon)

plt.figure()
plt.plot(np.arange(total_episodes), action_plot)
plt.title('Q-Learning Model')
plt.xlabel('Episodes')
plt.ylabel('Number of Actions')
plt.show()

    