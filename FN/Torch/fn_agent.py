from typing import Optional
import torch
import torch.nn as nn


class FNAgent():        
    model: Optional[nn.Module]
    def __init__(self, epsilon, actions):        
        epsilon = epsilon
        actions = actions
        self.model = None
        self.estimate_probs = False
        self.initialized = False
        
    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        
    @classmethod
    def load(cls, env, model_path, epsilon=0.0001):
        actions = list(range(env.action_space.n))
        agent = cls(epsilon, actions)
        agent.model = torch.load(model_path)
        agent.initialized = True
        return agent
    
    def initialize(self, experiences):
        pass
    
    def estimate(self, state):
        pass
    
    def update(self, experiences, gamma):
        pass
    
    def policy(self, state):
        pass
    
    def play(self, env, episode_count = 5, render = True):
        for episode in range(episode_count):
            state = env.reset()
            done = False
            episode_reward = 0
            while not done:
                if render:
                    env.render()
                action = self.policy(state)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state
            else:
                print("Get reward {}".format(episode_reward))