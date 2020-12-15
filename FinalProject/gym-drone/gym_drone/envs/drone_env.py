import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pygame
import numpy as np
import random
import math
class NeeveEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # In pygame a block is of size 100, hence to use divide by 100
        # Agent starting position in environment
        self.x = 100
        self.y = 500
        self.start_x = self.x
        self.start_y = self.y
        
        # Velocity of the agent
        self.vel = 100
        
        # Initialize the background map
        self.screen_size = [1000, 1000]
        self.observation_space = np.zeros((150, 4))
        self.nb_actions = 4
        self.map = pygame.display.set_mode(self.screen_size)
        self.backdrop = pygame.image.load('white_background.jpg').convert()
        self.backdrop = pygame.transform.scale(self.backdrop, self.screen_size)
        self.perimeter = self.map.get_rect()
        
        # Block size and position
        self.block_size = [100, 100]
        self.block = pygame.image.load('block.jpg').convert()
        self.block = pygame.transform.scale(self.block, self.block_size)
        self.num_blocks = 30
        self.count_blocks = 0
        self.block_pos = []
        self.block_states = []
        # Generate random initial block statesstates
        while self.count_blocks < self.num_blocks:
            self.rand_state = random.randint(1, int(self.screen_size[0]/100-2)), \
                                random.randint(0, int(self.screen_size[1]/100))
            if self.rand_state != (self.start_x, self.start_y):
                if self.rand_state not in self.block_states:
                    self.block_pos.append(self.block.get_rect().move(self.rand_state[0]*100, \
                                                                     self.rand_state[1]*100))
                    self.block_states.append(self.rand_state)
                    self.count_blocks += 1
            
            
        # Finish line
        self.finish_line_size = [100, 100]
        self.finish_line = pygame.image.load('finish_line.png').convert()
        self.finish_line = pygame.transform.scale(self.finish_line, self.finish_line_size)
        self.finish_line_pos = []
        for i in range(int(self.screen_size[1]/100)):
            self.finish_line_pos.append(self.block.get_rect().move( \
                                            self.screen_size[0]-100, i*100))
        
        # Initialize Agent height & width
        self.agent_width = 100
        self.agent_height = 100
        
        # Starting direction is 'Right'
        self.agent_img = pygame.image.load('drone.png').convert_alpha()
        self.agent = pygame.transform.scale(self.agent_img, 
                                             (self.agent_width, 
                                              self.agent_height))
        # Get current position of the agent
        self.position = self.agent.get_rect()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    # Determines the action to take by the agent
    def step(self, action):
        done = False
        wall_hit = False
        # Up
        if action == 0:
            self.y = self.y - self.vel
            block_hit = self.block_check()
            wall_hit = self.bounds_check()
            if wall_hit | block_hit:
                self.y = self.y + self.vel
        # Right
        elif action == 1:
            self.x = self.x + self.vel
            block_hit = self.block_check()
            wall_hit = self.bounds_check()
            if wall_hit | block_hit:
                self.x = self.x - self.vel
        # Down
        elif action == 2:
            self.y = self.y + self.vel
            block_hit = self.block_check()
            wall_hit = self.bounds_check()
            if wall_hit | block_hit:
                self.y = self.y - self.vel
        # Left
        elif action == 3:
            self.x = self.x - self.vel
            block_hit = self.block_check()
            wall_hit = self.bounds_check()
            if wall_hit | block_hit:
                self.x = self.x + self.vel
        distance = self.x/100
        
        if wall_hit | block_hit:
            reward = -1
        else:
            reward = distance * 0.1
        # Finish line
        if self.x >= self.screen_size[0]-100:
            reward = 10
            done = True
            
        # Update new position
        self.position = self.agent.get_rect().move(self.x, self.y)
        self.state = (int(self.x/100), int(self.y/100))
        observations = [self.state, reward, done, {}]
        
        return observations
    
    # Checks to see if hit one of the randomly generated blocks
    def block_check(self):
        if (int(self.x/100), int(self.y/100)) in self.block_states:
            return True
        else:
            return False
    
    # Checks if a wall was hit
    def bounds_check(self):
        walls = [0, 0, self.screen_size[0], self.screen_size[1]]
        if (self.x < walls[0]) | (self.x > walls[2]) | \
                (self.y < walls[1]) | (self.y > walls[3]):
            return True
        else:
            return False
            
        
    # Restarts map in agents original starting positions
    def reset(self):
        self.x, self.y = self.start_x, self.start_y
        self.map.blit(self.backdrop, self.perimeter)
        for i in range(len(self.block_pos)):
            self.map.blit(self.block, self.block_pos[i])
        for i in range(len(self.finish_line_pos)):
            self.map.blit(self.finish_line, self.finish_line_pos[i])
        self.map.blit(self.agent, self.position)
        pygame.display.update() 
    
    # Updates the map with new agent, obstacles, and environment positions
    def render(self, mode='human'):	
        self.map.blit(self.backdrop, self.perimeter)
        for i in range(len(self.block_pos)):
            self.map.blit(self.block, self.block_pos[i])
        for i in range(len(self.finish_line_pos)):
            self.map.blit(self.finish_line, self.finish_line_pos[i])
        self.map.blit(self.agent, self.position)
        pygame.display.update() 
    
    # This closes the pygame loaded up
    def close(self):
        pygame.quit()
