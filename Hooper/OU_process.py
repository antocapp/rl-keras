import numpy as np
import random

class OU_old:
    
    def __init__(self, action_dim, mu=0, theta=0.3, sigma=0.15):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        
    def noise(self, x):
        return self.theta*(self.mu - x) + self.sigma*np.random.randn(self.action_dim)

class OU(object):

    def function(self, x, mu, theta=0.15, sigma=0.3):
        return theta * (mu - x) + sigma * np.random.randn(1)

class OrnsteinUhlenbeckNoise:
  # Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py

  def __init__(self, action_dim, theta=0.15, sigma=0.3, mu=0):
    self.action_dim = action_dim

    self.mu = mu
    self.theta = theta
    self.sigma = sigma

    self.state = np.ones(self.action_dim) * self.mu
    self.reset()

  def add_noise(self):
    x = self.state
    dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
    self.state = x + dx

    return self.state

  def reset(self):
    self.state = np.ones(self.action_dim) * self.mu