import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pygame
import numpy as np

class NeeveEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Agent starting position in environment
        self.x = 100
        self.y = 600
        self.start_x = self.x
        self.start_y = self.y
        
        # Velocity of the agent
        self.vel = 100
        
        # Initialize the background map
        self.screen_size = [1500, 1000]
        self.map = pygame.display.set_mode(self.screen_size)
        self.backdrop = pygame.image.load('bamboo_house.jpg').convert()
        self.backdrop = pygame.transform.scale(self.backdrop, self.screen_size)
        self.perimeter = self.map.get_rect()
        
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
        reward = 1
        done = False
        wall_hit = False
        # Up
        if action == 0:
            self.y = self.y - self.vel
            wall_hit = self.bounds_check()
            if wall_hit:
                self.y = self.y + self.vel
        # Right
        elif action == 1:
            self.x = self.x + self.vel
            wall_hit = self.bounds_check()
            if wall_hit:
                self.x = self.x - self.vel
        # Down
        elif action == 2:
            self.y = self.y + self.vel
            wall_hit = self.bounds_check()
            if wall_hit:
                self.y = self.y - self.vel
        # Left
        elif action == 3:
            self.x = self.x - self.vel
            wall_hit = self.bounds_check()
            if wall_hit:
                self.x = self.x + self.vel
        if wall_hit:
            reward = 0
            
        if self.x >= 1400:
            reward = 1
            done = True
        # Update new position
        self.position = self.agent.get_rect().move(self.x, self.y)
        self.state = (int(self.x/10), int(self.y/10))
        observations = [self.state, reward, done, {}]
        return observations
    
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
        self.map.blit(self.agent, self.position)
        pygame.display.update() 
    
    # Updates the map with new agent, obstacles, and environment positions
    def render(self, mode='human'):	
        self.map.blit(self.backdrop, self.perimeter)
        self.map.blit(self.agent, self.position)
        pygame.display.update() 
    
    # This closes the pygame loaded up
    def close(self):
        pygame.quit()
